import dataclasses
import shutil
import time
from pathlib import Path

from ..common.io import write_text
from ..common.runlog import log
from ..common.text import fmt_hours, md_table
from .config import FilterSettings, GenSettings, MiningSettings, TrainSettings

XLMR_LARGE_BODY = 303e6


def a100_estimates(stage, data):
    e, timings = stage.cfg.estimate, stage.timings
    mining, t2 = data.stats2.get("mining") or {}, data.stats2.get("timings") or {}
    body = sum(p.numel() for n, p in stage.encoder.named_parameters() if "embeddings" not in n)
    real = not stage.smoke
    n_pass = GenSettings().n_passages
    pool_n = (sum(v["passages"] for v in mining["pools"].values()) if real and mining.get("pools")
              else 2_064_310 + MiningSettings().news_pool_size + 8_841_823)
    n_q = mining.get("rerank", {}).get("queries", 0) if real else 1_750_000
    rows = sum(mining.get("train_rows", {}).values()) if real else 1_080_000
    gen = data.stats1.get("generation") or {}
    if gen.get("passages"):
        ptok, ctok = gen["prompt_tokens"] / gen["passages"], gen["completion_tokens"] / gen["passages"]
    else:
        gt = data.stats1.get("generation_test") or {"prompt_tokens": 550, "completion_tokens": 110}
        ptok, ctok = gt["prompt_tokens"], gt["completion_tokens"]
    gen_h = n_pass * (ptok / e.a100_gen_prompt_tokens_per_s + ctok / e.a100_gen_output_tokens_per_s) / 3600
    enc_t = t2.get("encode", {}).get("tokens_per_passage", 110)
    enc_h = 2 * body * enc_t * pool_n / (e.a100_encode_tflops * 1e12) / 3600
    rr = t2.get("rerank", {})
    pairs_q = 1.2 + MiningSettings().rerank_top
    rr_h = 2 * XLMR_LARGE_BODY * rr.get("tokens_per_pair", 150) * n_q * pairs_q / (e.a100_rerank_tflops * 1e12) / 3600
    tok_q = timings["train"]["tokens_per_step"] / timings["train"]["batch_size"]
    batch, eval_every = TrainSettings().batch_size, TrainSettings().eval_every
    steps = rows // batch
    train_h = 8 * body * tok_q * batch * steps / (e.a100_train_tflops * 1e12) / 3600
    val_corpus = (len(stage.source_names) * FilterSettings().val_queries_per_source * MiningSettings().val_candidates
                  + MiningSettings().val_random_passages)
    val_h = (steps // eval_every + 1) * 2 * body * enc_t * val_corpus / (e.a100_encode_tflops * 1e12) / 3600
    a100 = {"generation (Qwen3.6-35B-A3B FP8, vLLM)": gen_h,
            f"pool encoding ({pool_n / 1e6:.1f}M passages)": enc_h,
            f"reranking ({n_q / 1e6:.2f}M queries x {pairs_q:.1f} pairs)": rr_h,
            f"training ({steps:,} steps of {batch})": train_h,
            f"validation during training ({steps // eval_every + 1} x ~{val_corpus / 1e3:.0f}k passages)": val_h}
    lines = [f"{'step':<62}{'A100 80GB':>11}"]
    lines += [f"{k:<62}{fmt_hours(v * 3600):>11}" for k, v in a100.items()]
    lines.append(f"{'total (evaluation: the evaluation stage, on your PC)':<62}{fmt_hours(sum(a100.values()) * 3600):>11}")
    lines.append(f"measured: generation {ptok:.0f} prompt + {ctok:.0f} output tokens per passage; pool passages "
                 f"{enc_t:.0f} tokens; reranker pairs {rr.get('tokens_per_pair', 150):.0f} XLM-R tokens; training "
                 f"{tok_q:.0f} tokens per query with its {data.n_neg + 1} passages; assumed A100 speeds: "
                 f"{dataclasses.asdict(e)}")
    log("\n".join(lines))
    timings["a100_hours"] = a100
    return a100


def stage_report(stage, data, summary, exported, before_val, tests, a100):
    cfg, s, names = stage.cfg, stage.cfg.train, stage.source_names
    f1 = data.stats1.get("funnel", {})
    mining = data.stats2.get("mining") or {}
    mix = mining.get("final_mix", {})
    steps = list(f1)
    stage3 = Path(stage.stage3_dir)
    root = stage.project.root
    shown_stage3 = stage3.relative_to(root).as_posix() if stage3.is_relative_to(root) else stage3

    def val_row(label, m):
        return [label, m.get("step"), *[f"{m[k]:.4f}" if m.get(k) is not None else "-"
                                        for k in ("ndcg@10", "recall@10", "recall@100")]]

    consistency = mining.get("consistency", {})
    return "\n\n".join([
        f"# Stage 4: supervised fine-tuning with hard negatives{' (SMOKE RUN)' if stage.smoke else ''}",
        f"`pired.supervised`, {time.strftime('%Y-%m-%d')}. Initialized from the stage-3 model "
        f"(`{shown_stage3}`). Mean pooling + L2 norm; prefixes `{stage.q_prefix}` / `{stage.p_prefix}`; limits "
        f"{stage.max_q} / {stage.max_p} tokens.",
        "**The train splits of MIRACL and Mr.TyDi were used for training**: MIRACL-ar and Mr.TyDi-ar are in-domain "
        "from this stage on. Dev and test queries of every Arabic mteb retrieval task were removed from the training "
        "data. The mteb evaluation is the evaluation stage (`reports/evaluation.md` of stage 5).",
        "## Data (Part 1)", "Details: `stage_supervised_data.md` (same folder).",
        md_table(["source", *steps], [[src, *[f1[st].get(src, 0) for st in steps]] for src in names])
        if f1 else "(Part 1 statistics not found)",
        "## Final mix (Part 2)",
        md_table(["source", "available", "final queries", "share", "training rows", "validation", "license"],
                 [[src, v["available"], v["final"], f"{v['share']:.1%}", v["rows"], v["validation"], v["license"]]
                  for src, v in mix.items()]) if mix else "",
        "## Mining (Part 2)",
        md_table(["source", "reranked candidates", "false negatives removed"],
                 [[src, v["candidates_scored"], v["false_negatives"]]
                  for src, v in mining.get("false_negatives", {}).items()]),
        md_table(["source", "queries", "positive ranked below the cutoff", "action"],
                 [[src, v["queries"], v.get("rank_above_cutoff", v.get("rank_above_3")), v["action"]]
                  for src, v in consistency.items()]),
        f"Synthetic questions cut to 2 per passage: {mining.get('synthetic_cut_to_2_per_passage', 0):,}. Negatives: "
        + ", ".join(f"{k} {v:,}" for k, v in mining.get("negatives_origin", {}).items()) + ".",
        "## Training (Part 3)",
        md_table(["setting", "value"], [
            ["batch", f"{s.batch_size} queries x {1 + data.n_neg} passages, one source per batch (GradCache)"],
            ["loss", f"InfoNCE with {data.n_neg} hard negatives + in-batch negatives, temperature {s.temperature}, "
                     f"reverse direction {s.bidirectional}; own negatives scored above {s.max_neg_score} by the "
                     "reranker ignored for their query"],
            ["optimizer", f"AdamW lr {s.lr}, wd {s.weight_decay}, warmup {s.warmup_fraction:.0%}, linear decay, clip "
                          f"{s.max_grad_norm}"],
            ["steps", f"{summary['steps']:,} ({summary['queries_seen']:,} queries, {summary['tokens_seen']:,} tokens, "
                      f"{summary['hours']:.2f} h)"]]),
        "Validation (held-out queries searched in their corpus: positives + mined candidates + random pool passages):",
        md_table(["step", "val nDCG@10", "val R@10", "val R@100", "cos unrelated q-p", "cos unrelated p-p"],
                 [[r["step"], f"{r['ndcg@10']:.4f}", f"{r['recall@10']:.4f}", f"{r['recall@100']:.4f}",
                   f"{r['cos_unrelated_qp']:.3f}", f"{r['cos_unrelated_pp']:.3f}"] for r in exported["validations"]]),
        md_table(["exported model", "step", "val nDCG@10", "val R@10", "val R@100"],
                 [val_row("before this stage (stage 3)", {**before_val, "step": 0}),
                  val_row(f"{cfg.hub.best_name} (chosen)", exported["best"]),
                  val_row(cfg.hub.final_name, exported["final"])]),
        "## Tests", md_table(["test", "result"], [[k, v] for k, v in tests.items()]),
        "## Estimated time of the real run (A100 80GB, without evaluation)",
        md_table(["step", "estimate"], [[k, fmt_hours(v * 3600)] for k, v in a100.items()]),
    ])


def write_reports_and_push(stage, report):
    cfg = stage.cfg
    export_dir = Path(cfg.final_dir)
    write_text(report, cfg.reports_dir / "stage_supervised.md")
    write_text(report, export_dir / "reports" / "stage_supervised.md")
    data_report = cfg.reports_dir / "stage_supervised_data.md"
    if data_report.exists():
        shutil.copyfile(data_report, export_dir / "reports" / "stage_supervised_data.md")
    write_text(f"# {stage.project.name}: stage 4 (supervised with hard negatives){' - SMOKE RUN' if stage.smoke else ''}"
               f"\n\n- `{cfg.hub.best_name}/`: the checkpoint with the best validation nDCG@10 (the chosen model)\n"
               f"- `{cfg.hub.final_name}/`: the last step of training\n"
               "- `run_logs/`, `reports/`: the logs and reports of stage 4\n", export_dir / "README.md")
    stage.push_folder(export_dir, stage.repo_id("model"), "model")
