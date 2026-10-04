import hashlib
import json
import random
import shutil
import time
from collections import Counter

import numpy as np
import pandas as pd

from ..common.hub import push_folder
from ..common.io import read_json, write_parquet, write_text
from ..common.runlog import log
from ..common.text import md_table, short
from ..tokenizer.fingerprint import vocab_fingerprint
from .steps import STEP_ORDER, clean_shards, format_removed, load_alive, load_samples, step_fingerprint, summary_of


def final_dir(stage):
    return stage.work / "final"


def capped_counts(stage, available):
    n = {s: available[s] if (stage.smoke or stage.sources[s].target is None) else min(available[s],
                                                                                        stage.sources[s].target)
         for s in available}
    share = stage.cfg.filters.max_share
    while True:
        total = sum(n.values())
        over = [s for s in n if n[s] > share * total]
        if not over:
            return n
        for s in over:
            n[s] = int(share / (1 - share) * (total - n[s]))


def run_final_mix(stage):
    cfg, final = stage.cfg, final_dir(stage)
    fingerprint = hashlib.sha256((step_fingerprint(stage, "consistency") + json.dumps(
        [cfg.filters.max_share, cfg.filters.val_pairs_per_source, [(s.name, s.target) for s in cfg.data.sources]])
                                  ).encode()).hexdigest()[:16]
    manifest_file = final / "manifest.json"
    if manifest_file.exists() and json.loads(manifest_file.read_text(encoding="utf-8"))["fingerprint"] == fingerprint:
        log(f"final mix: done earlier, loaded from {manifest_file}")
        return json.loads(manifest_file.read_text(encoding="utf-8"))
    shutil.rmtree(final, ignore_errors=True)
    available = {s: int(load_alive(stage, "consistency", s).sum()) for s in stage.source_names}
    counts = capped_counts(stage, available)
    sources = {}
    for si, s in enumerate(stage.source_names):
        rng = np.random.default_rng([cfg.data.seed, si, 9])
        chosen = rng.permutation(np.flatnonzero(load_alive(stage, "consistency", s)))[: counts[s]]
        n_val = min(cfg.filters.val_pairs_per_source, len(chosen) // 5)
        val_ids = set(chosen[:n_val].tolist())
        chosen_mask = np.zeros(summary_of(stage)[s]["pairs"], dtype=bool)
        chosen_mask[chosen] = True
        n_train, val_parts = 0, []
        for path in clean_shards(stage, s):
            df = pd.read_parquet(path, columns=["pair_id", "query", "passage"])
            df = df[chosen_mask[df["pair_id"].to_numpy()]]
            is_val = df["pair_id"].isin(val_ids).to_numpy()
            train = df.loc[~is_val, ["query", "passage"]].reset_index(drop=True)
            train.insert(0, "source", s)
            if len(train):
                write_parquet(train, final / "train" / s / f"part-{path.stem.split('-')[1]}.parquet")
            n_train += len(train)
            val_parts.append(df[is_val])
        val = pd.concat(val_parts).set_index("pair_id").loc[chosen[:n_val], ["query", "passage"]].reset_index(drop=True)
        val.insert(0, "source", s)
        write_parquet(val, final / "validation" / s / "part-00000.parquet")
        sources[s] = {"available": available[s], "target": stage.sources[s].target, "final": int(counts[s]),
                      "train": n_train, "validation": len(val), "license": stage.sources[s].license}
    total = sum(v["final"] for v in sources.values())
    for v in sources.values():
        v["share"] = round(v["final"] / max(total, 1), 4)
    manifest = {"fingerprint": fingerprint, "columns": ["source", "query", "passage"], "sources": sources,
                "total": total, "prefixes": {"query": stage.q_prefix, "passage": stage.p_prefix},
                "max_tokens": {"query": stage.max_q, "passage": stage.max_p},
                "tokenizer_fingerprint": vocab_fingerprint(stage.tokenizer.get_vocab()),
                "smoke": stage.smoke, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


def format_mix(stage, manifest):
    lines = [f"{'source':<10}{'available':>11}{'target':>11}{'final':>10}{'share':>8}{'train':>10}{'validation':>12}"]
    for s, v in manifest["sources"].items():
        target = f"{v['target']:,}" if v["target"] and not stage.smoke else "all"
        lines.append(f"{s:<10}{v['available']:>11,}{target:>11}{v['final']:>10,}{v['share']:>8.1%}{v['train']:>10,}"
                     f"{v['validation']:>12,}")
    lines.append(f"{'total':<10}{sum(v['available'] for v in manifest['sources'].values()):>11,}{'':>11}"
                 f"{manifest['total']:>10,}")
    biggest = max(manifest["sources"].items(), key=lambda kv: kv[1]["share"])
    lines.append(f"largest share: {biggest[0]} {biggest[1]['share']:.1%} (cap {stage.cfg.filters.max_share:.0%})"
                 + ("   [SMOKE: targets are ignored, only the cap applies]" if stage.smoke else ""))
    if biggest[1]["share"] > stage.cfg.filters.max_share + 1e-9:
        raise AssertionError(f"{biggest[0]} is above the share cap")
    return "\n".join(lines)


def final_split(stage, split, source):
    files = sorted((final_dir(stage) / split / source).glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["source", "query", "passage"])
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def manual_review(stage, kept_per_source=100, removed_per_step=20):
    lines = []
    for s in stage.source_names:
        files = sorted((final_dir(stage) / "train" / s).glob("*.parquet"))
        pick = random.Random(0).sample(files, min(3, len(files)))
        df = pd.concat([pd.read_parquet(f) for f in pick], ignore_index=True) if pick else pd.DataFrame()
        lines.append(f"\n=== KEPT: {min(kept_per_source, len(df))} random pairs of {s} ===")
        for i, r in enumerate(df.sample(min(kept_per_source, len(df)), random_state=0).itertuples(), 1):
            lines.append(f"{i:>3}. Q: {short(r.query, 140)}")
            lines.append(f"     P: {short(r.passage, 240)}")
    for step in STEP_ORDER:
        samples = load_samples(stage, step)
        stats = json.loads((stage.steps_dir / step / "stats.json").read_text(encoding="utf-8"))
        n_removed = sum(sum(s["removed"].values()) for s in stats["sources"].values())
        n_shown = min(removed_per_step, len(samples.drop_duplicates(["query", "passage"])))
        lines.append(f"\n=== REMOVED by {step}: {n_shown} random distinct pairs (of {n_removed:,} removed) ===")
        lines.append(format_removed(samples, removed_per_step, seed=2))
    return "\n".join(lines)


def funnel_table(stage, manifest):
    columns = ["raw", *STEP_ORDER, "final", "train", "validation"]
    rows = {}
    for s in stage.source_names:
        steps = [json.loads((stage.steps_dir / st / "stats.json").read_text(encoding="utf-8"))["sources"][s]["out"]
                 for st in STEP_ORDER]
        m = manifest["sources"][s]
        rows[s] = [summary_of(stage)[s]["pairs"], *steps, m["final"], m["train"], m["validation"]]
    rows["total"] = [sum(v[i] for v in rows.values()) for i in range(len(columns))]
    return columns, rows


def data_report(stage, manifest, consistency_stats):
    cfg = stage.cfg
    loading_file = stage.raw / "loading.json"
    loading = stage.loading or (read_json(loading_file) if loading_file.exists() else {})
    columns, rows = funnel_table(stage, manifest)
    reasons = Counter()
    for step in STEP_ORDER:
        for src in json.loads((stage.steps_dir / step / "stats.json").read_text(encoding="utf-8"))["sources"].values():
            for k, n in src["removed"].items():
                reasons[(step, k)] += n
    return "\n\n".join([
        f"# Stage 3 data: weakly supervised pairs{' (SMOKE RUN: small subsets)' if stage.smoke else ''}",
        f"Built by `pired.contrastive` on {time.strftime('%Y-%m-%d')}. Columns: `source, query, passage`. "
        f"Prefixes added at training time: query `{stage.q_prefix}`, passage `{stage.p_prefix}`. Limits: query "
        f"{stage.max_q}, passage {stage.max_p} tokens (prefix and [CLS]/[SEP] included).",
        "## Sources",
        md_table(["source", "pairs", "loaded from", "license"],
                 [[s, stage.sources[s].description, loading.get(s, ""), stage.sources[s].license]
                  for s in stage.source_names]),
        "Wikipedia section headings: the `wikimedia/wikipedia` text keeps them as plain lines (markup removed), so "
        "both pair types (title → lead, \"title heading\" → section) are built. Mined Q&A comes from FineWeb-2 because "
        "the 101B dataset's text has no punctuation (no ؟ to find questions).",
        "## Funnel (pairs left after each step)",
        md_table(["source", *columns], [[s, *v] for s, v in rows.items()]),
        "## Removed, by step and reason",
        md_table(["step", "reason", "pairs"], [[st, k, n] for (st, k), n in reasons.items()]),
        "## bge-m3 rank of the true passage among 100",
        md_table(["source", "rank histogram", "kept at top-1/3/5/10"],
                 [[s, v["rank histogram"], v["kept at top-1/3/5/10"]] for s, v in consistency_stats["sources"].items()]),
        "## Final mix",
        md_table(["source", "available", "target", "final", "share", "train", "validation"],
                 [[s, v["available"], v["target"] or "all", v["final"], f"{v['share']:.1%}", v["train"],
                   v["validation"]] for s, v in manifest["sources"].items()]),
        f"No source above {cfg.filters.max_share:.0%} of the final set. MIRACL, Mr.TyDi and test splits were used only "
        "to decontaminate (D7).",
    ])


def write_data_report(stage, manifest, consistency_stats):
    report = data_report(stage, manifest, consistency_stats)
    write_text(report, stage.cfg.reports_dir / "stage_contrastive_weak_data.md")
    write_text(report, final_dir(stage) / "README.md")
    return report


def push_final_data(stage):
    return push_folder(stage.api, final_dir(stage), stage.repo_id("data"), "dataset", stage.cfg.hub.push)
