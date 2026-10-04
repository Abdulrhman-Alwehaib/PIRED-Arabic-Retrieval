import json
import math
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from ..common.io import read_json, write_text
from ..common.runlog import log, read_jsonl
from ..common.text import md_table
from ..encoder.configuration import ArabicEncoderConfig
from ..encoder.modeling import ArabicEncoderModel
from .mix import funnel_table
from .run import load_eval_results, results_header, results_table, run_dir
from .sources import raw_summary
from .steps import STEP_ORDER


def collect_timings(stage):
    timings = {}
    for name, summary in raw_summary(stage).items():
        timings[f"build_{name}"] = {"seconds": summary["unit_seconds"], "rows": summary["made"], "workers": 1}
    for step in STEP_ORDER:
        f = stage.steps_dir / step / "stats.json"
        if f.exists():
            stats = read_json(f)
            timings[step] = {"seconds": stats["seconds"], "rows": sum(s["in"] for s in stats["sources"].values()),
                             "workers": stats["workers"]}
            if "gpu" in stats:
                timings["consistency_gpu"] = stats["gpu"]
    train_file = run_dir(stage) / "train_summary.json"
    if train_file.exists():
        timings["train"] = read_json(train_file)
    return timings


def repo_bytes(stage, repo, prefix="", revision=None, repo_type="dataset"):
    api = stage.api
    info = (api.dataset_info if repo_type == "dataset" else api.model_info)(repo, revision=revision, files_metadata=True)
    return {s.rfilename: s.size or 0 for s in info.siblings if s.rfilename.startswith(prefix)}


def repo_rows(stage, repo, files, revision=None):
    out = []
    for f in files:
        with stage.fs.open(stage.remote_path(repo, f, revision), "rb") as fh:
            out.append(pq.ParquetFile(fh).metadata.num_rows)
    return out


def body_parameters(stage):
    with torch.device("meta"):
        model = ArabicEncoderModel(ArabicEncoderConfig.from_pretrained(stage.mlm_dir))
    return sum(p.numel() for n, p in model.named_parameters() if not n.startswith("embeddings.tok_embeddings"))


def estimate_real_run(stage, manifest):
    cfg, e = stage.cfg, stage.cfg.estimate
    d = cfg.data
    summary = raw_summary(stage)
    timings = collect_timings(stage)
    st = {s: summary[s]["stats"] for s in stage.source_names}
    per_doc = {"news": summary["news"]["made"] / st["news"]["articles"],
               "wiki": summary["wiki"]["made"] / st["wiki"]["articles"],
               "qa_mined": summary["qa_mined"]["made"] / st["qa_mined"]["documents"],
               "spans": summary["spans"]["made"] / st["spans"]["documents"]}
    news_files = stage.repo_parquet_files(d.news_repo, "data")
    wiki_files = stage.repo_parquet_files(d.wiki_repo, d.wiki_prefix)
    fineweb_files = stage.repo_parquet_files(d.fineweb2_repo, d.fineweb2_prefix)
    b101_files = stage.repo_parquet_files(d.b101_repo, d.b101_prefix)
    fw_rows = repo_rows(stage, d.fineweb2_repo, fineweb_files[:3])
    b101_rows = repo_rows(stage, d.b101_repo, b101_files[:5])
    span_fw = d.spans_corpus == "fineweb2"
    span_rows, span_files_all = (fw_rows, fineweb_files) if span_fw else (b101_rows, b101_files)
    qa = st["qa"]
    available = {
        "news": per_doc["news"] * sum(repo_rows(stage, d.news_repo, news_files)),
        "wiki": per_doc["wiki"] * sum(repo_rows(stage, d.wiki_repo, wiki_files)),
        "xlsum": 2 * sum(repo_rows(stage, d.xlsum_repo, [d.xlsum_file], d.xlsum_revision)),
        "qa": (qa.get("found in MRC/train.json", 0) - sum(v for k, v in qa.items() if k.startswith("skipped"))
               + sum(v for k, v in qa.items() if k.endswith(" rows") and not k.startswith("rows"))),
        "qa_mined": per_doc["qa_mined"] * np.mean(fw_rows) * len(fineweb_files),
        "spans": per_doc["spans"] * np.mean(span_rows) * len(span_files_all)}
    full_raw = {s: int(min(available[s], stage.sources[s].raw or available[s])) for s in stage.source_names}
    raw_total = sum(full_raw.values())
    decontam = read_json(stage.steps_dir / "decontam" / "stats.json")["sources"]
    survive = {s: v["out"] / max(summary[s]["pairs"], 1) for s, v in decontam.items()}
    to_bge = sum(full_raw[s] * survive[s] for s in stage.source_names)
    lines = ["raw pairs of the real run (budget, or all a source has): "
             + ", ".join(f"{s} {v / 1e6:.2f}M" for s, v in full_raw.items()) + f" = {raw_total / 1e6:.1f}M",
             f"with the smoke survival rates up to D7 ({', '.join(f'{s} {v:.0%}' for s, v in survive.items())}), about "
             f"{to_bge / 1e6:.1f}M pairs reach bge-m3"]
    fw_files = math.ceil(full_raw["qa_mined"] / (per_doc["qa_mined"] * np.mean(fw_rows)))
    span_files = math.ceil(full_raw["spans"] / (per_doc["spans"] * np.mean(span_rows)))
    fw_bytes = repo_bytes(stage, d.fineweb2_repo, d.fineweb2_prefix + "/")
    b101_bytes = repo_bytes(stage, d.b101_repo, "data/")
    downloads = {"news": sum(repo_bytes(stage, d.news_repo, "data/").values()),
                 "wiki": sum(repo_bytes(stage, d.wiki_repo, d.wiki_prefix + "/").values()),
                 f"FineWeb-2 ({fw_files} files for mined Q&A)": fw_files * np.mean(list(fw_bytes.values())),
                 f"{'FineWeb-2' if span_fw else '101B'} ({span_files} files for spans)":
                     span_files * np.mean(list((fw_bytes if span_fw else b101_bytes).values())),
                 "bge-m3, e5-base, MIRACL + Mr.TyDi corpora": 2.3e9 + 1.1e9 + 1.1e9}
    lines.append("downloads: " + ", ".join(f"{k} {v / 1e9:.1f} GB" for k, v in downloads.items())
                 + f" -> {sum(downloads.values()) / 1e9:.0f} GB, ~{sum(downloads.values()) / 1e6 / e.download_mb_per_s / 60:.0f}"
                 f" min at {e.download_mb_per_s:.0f} MB/s")
    smoke_raw = sum(summary[s]["pairs"] for s in stage.source_names)
    cpu_rows = []
    for step, t in timings.items():
        if step.startswith("build_") or step in STEP_ORDER[:-1]:
            per_row = t["seconds"] * t["workers"] / max(t["rows"], 1)
            full_rows = full_raw[step[6:]] if step.startswith("build_") else raw_total * t["rows"] / smoke_raw
            cpu_rows.append((step, per_row * 1e3, per_row * full_rows / e.cpu_cores / 60))
    lines.append(f"CPU steps (ms per row here; x rows of the real run / {e.cpu_cores} cores):")
    lines += [f"  {step:<16}{ms:>8.2f} ms/row  -> ~{minutes:5.0f} min" for step, ms, minutes in cpu_rows]
    cpu_total = sum(m for _, _, m in cpu_rows)
    lines.append(f"  total CPU ~{cpu_total / 60:.1f} h")
    g = timings.get("consistency_gpu")
    bge_hours = float("nan")
    if g:
        bge_tok_per_pair = g["tokens"] / g["pairs"]
        bge_hours = 2 * 303e6 * bge_tok_per_pair * to_bge / (e.h100_infer_tflops * 1e12) / 3600
        lines.append(f"bge-m3: {bge_tok_per_pair:.0f} XLM-R tokens per pair; here {g['tokens'] / g['seconds']:,.0f} "
                     f"tokens/s; real run {to_bge / 1e6:.1f}M pairs -> ~{bge_hours:.1f} h on an H100 at "
                     f"{e.h100_infer_tflops:.0f} TFLOP/s")
    else:
        lines.append("bge-m3: no scoring was measured, no estimate")
    train = timings["train"]
    n_body = body_parameters(stage)
    tok_per_pair = train["tokens"] / (train["steps"] * train["batch_size"])
    final_pairs = sum(v["train"] for v in manifest["sources"].values()) if not stage.smoke else 13_000_000 - 30_000
    full_steps = final_pairs // 4096
    step_flops = 8 * n_body * tok_per_pair * 4096
    step_s = step_flops / (e.h100_train_tflops * 1e12)
    lines.append(f"training: {tok_per_pair:.0f} tokens per pair; here {train['tokens'] / train['seconds']:,.0f} tokens/s "
                 f"(batch {train['batch_size']}, peak memory {train['peak_mem_gb']:.1f} GiB)")
    lines.append(f"  full run: {final_pairs / 1e6:.1f}M pairs / 4096 = {full_steps:,} steps x {step_flops / 1e15:.2f} "
                 f"PFLOP = ~{step_s:.1f} s/step at {e.h100_train_tflops:.0f} TFLOP/s -> "
                 f"~{full_steps * step_s / 3600:.1f} h; save_every for a checkpoint every ~30 min: "
                 f"~{max(1, round(1800 / step_s)):,} steps")
    estimates = {"download_gb": round(sum(downloads.values()) / 1e9), "cpu_hours": round(cpu_total / 60, 1),
                 "bge_hours": round(bge_hours, 1), "train_hours": round(full_steps * step_s / 3600, 1),
                 "train_steps": full_steps, "save_every_30min": max(1, round(1800 / step_s)),
                 "pairs_to_bge_m": round(to_bge / 1e6, 1)}
    log("\n".join(lines))
    return estimates


def stage_report(stage, manifest, estimates=None):
    cfg, s = stage.cfg, stage.cfg.train
    records = read_jsonl(run_dir(stage) / "train_log.jsonl")
    val_records = [r for r in records if r.get("event") == "validation"]
    train_rows = [r for r in records if "loss" in r and "event" not in r]
    train = read_json(run_dir(stage) / "train_summary.json")
    tests_file = run_dir(stage) / "test_results.json"
    tests = read_json(tests_file) if tests_file.exists() else {}
    pooling = read_json(cfg.final_dir / "pooling_config.json")
    columns, rows = funnel_table(stage, manifest)
    parts = [
        f"# Stage 3: weakly supervised contrastive training{' (SMOKE RUN: small subsets, 20 steps)' if stage.smoke else ''}",
        f"`pired.contrastive`, {time.strftime('%Y-%m-%d')}. Encoder initialized from the final MLM model "
        f"(`{pooling['initialized_from']}`), MLM head dropped. Mean pooling over real tokens + L2 norm; prefixes "
        f"`{stage.q_prefix}` / `{stage.p_prefix}`; limits {stage.max_q} / {stage.max_p} tokens.",
        "## Data", "Details: `stage_contrastive_weak_data.md` (same folder).",
        md_table(["source", *columns], [[src, *v] for src, v in rows.items()]),
        md_table(["source", "final", "share", "train", "validation", "license"],
                 [[src, v["final"], f"{v['share']:.1%}", v["train"], v["validation"], v["license"]]
                  for src, v in manifest["sources"].items()]),
        "## Training",
        md_table(["setting", "value"], [
            ["batch", f"{s.batch_size:,} pairs, one source per batch (GradCache)"],
            ["loss", f"InfoNCE, temperature {s.temperature}, both directions: {s.bidirectional}"],
            ["optimizer", f"AdamW lr {s.lr}, wd {s.weight_decay}, warmup {s.warmup_fraction:.0%}, linear decay, clip "
                          f"{s.max_grad_norm}"],
            ["steps", f"{train['steps']:,} ({train['pairs_seen']:,} pairs, {train['tokens']:,} tokens, "
                      f"{train['seconds'] / 3600:.2f} h)"],
            ["precision", "bf16 autocast, fp32 weights and optimizer, loss in fp32 (TF32 off)"]]),
        md_table(["step", "val nDCG@10", "val R@10", "cos unrelated q-p", "cos unrelated p-p"],
                 [[r["step"], f"{r['ndcg@10']:.4f}", f"{r['recall@10']:.4f}", f"{r['cos_unrelated_qp']:.3f}",
                   f"{r['cos_unrelated_pp']:.3f}"] for r in val_records]),
    ]
    if train_rows:
        parts.append(f"Train loss {train_rows[0]['loss']:.3f} → {train_rows[-1]['loss']:.3f}; in-batch accuracy "
                     f"{train_rows[0]['acc_qp']:.2f} → {train_rows[-1]['acc_qp']:.2f}.")
    parts += ["## Evaluation (nDCG@10, Recall@100)",
              md_table(results_header(stage), results_table(stage, load_eval_results(stage))),
              ("SMOKE: MIRACL and Mr.TyDi were cut to 200 queries and ~5k documents each; the numbers only check the "
               "code." if stage.smoke else "MIRACL-ar (dev) and Mr.TyDi-ar (test) via mteb, full corpora; reported "
                                            "only, never used for decisions."),
              "## Tests", md_table(["test", "result"], [[k, v] for k, v in tests.items()])]
    if estimates:
        e = cfg.estimate
        parts += ["## Estimated time of the real run (one H100)",
                  md_table(["part", "estimate"], [
                      ["downloads", f"~{estimates['download_gb']} GB"],
                      ["CPU data steps", f"~{estimates['cpu_hours']} h on {e.cpu_cores} cores"],
                      ["bge-m3 filter", f"~{estimates['bge_hours']} h ({estimates['pairs_to_bge_m']}M pairs)"],
                      ["training", f"~{estimates['train_hours']} h ({estimates['train_steps']:,} steps)"]])]
    report = "\n\n".join(parts)
    write_text(report, cfg.reports_dir / "stage_contrastive_weak.md")
    write_text(json.dumps(estimates or {}, indent=1), run_dir(stage) / "estimates.json")
    return report
