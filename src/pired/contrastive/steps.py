import hashlib
import html
import json
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import regex

from ..common.io import write_parquet
from ..common.parallel import fork_available, parallel_map
from ..common.runlog import log
from ..common.text import short
from .config import EVAL_QUERY_FILES
from .context import active
from .sources import raw_shards, raw_summary

STEP_ORDER = ["cleanup", "length", "language", "junk", "near_identical", "dedupe", "decontam", "consistency"]
SAMPLE_PER_REASON = 30
SAMPLE_COLUMNS = ["pair_id", "kind", "query", "passage", "reason", "detail"]
JOB = {}


def step_settings(stage, step):
    f = stage.cfg.filters
    return {
        "cleanup": {"prefixes": [stage.q_prefix, stage.p_prefix], "max_tokens": [f.max_query_tokens, f.max_passage_tokens]},
        "length": {"min_tokens": [f.min_query_tokens, f.min_passage_tokens]},
        "language": {"min_arabic_letters": f.min_arabic_letters},
        "junk": {"boilerplate": list(f.boilerplate_titles), "max_query_repeats": f.max_query_repeats},
        "near_identical": {"max_jaccard": f.max_jaccard},
        "dedupe": {k: getattr(f, k) for k in ("minhash_ngram", "minhash_perms", "minhash_bands", "near_dup_jaccard")},
        "decontam": {"files": EVAL_QUERY_FILES, "ngram": f.decontam_ngram, "min_words": f.decontam_min_words},
        "consistency": {k: getattr(f, k) for k in ("bge_model", "bge_negatives", "bge_keep_top", "bge_query_max_len",
                                                   "bge_passage_max_len")},
    }[step]


def stable_hash(text):
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little", signed=True)


def clean_shards(stage, source):
    return sorted((stage.clean / source).glob("shard-*.parquet"))


def previous_step(step):
    i = STEP_ORDER.index(step)
    return STEP_ORDER[i - 1] if i else None


def summary_of(stage):
    if not hasattr(stage, "raw_summary"):
        stage.raw_summary = raw_summary(stage)
    return stage.raw_summary


def load_alive(stage, step, source):
    if step is None:
        return np.ones(summary_of(stage)[source]["pairs"], dtype=bool)
    return np.load(stage.steps_dir / step / f"{source}.alive.npy")


def step_fingerprint(stage, step):
    prev = previous_step(step)
    if prev:
        before = json.loads((stage.steps_dir / prev / "stats.json").read_text(encoding="utf-8"))
        before = before["fingerprint"] + json.dumps(before["sources"], sort_keys=True)
    else:
        summary = summary_of(stage)
        before = json.dumps({s: [summary[s]["pairs"], summary[s].get("units_key")] for s in stage.source_names})
    settings = json.dumps(step_settings(stage, step), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256((before + settings).encode("utf-8")).hexdigest()[:16]


def step_done(stage, step):
    f = stage.steps_dir / step / "stats.json"
    if f.exists():
        stats = json.loads(f.read_text(encoding="utf-8"))
        if stats["fingerprint"] == step_fingerprint(stage, step):
            log(f"{step}: done earlier, loaded from {f.parent} (delete that folder to run it again)")
            return stats
    return None


def fetch_texts(stage, source, ids, columns=("kind", "query", "passage")):
    ids = np.asarray(ids, dtype=np.int64)
    if not len(ids):
        return pd.DataFrame(columns=["pair_id", *columns])
    starts = np.asarray(summary_of(stage)[source]["shard_starts"])
    shard_of = np.searchsorted(starts, ids, side="right") - 1
    shards, parts = clean_shards(stage, source), []
    for k in np.unique(shard_of):
        df = pd.read_parquet(shards[k], columns=["pair_id", *columns])
        parts.append(df[df["pair_id"].isin(ids[shard_of == k])])
    return pd.concat(parts).set_index("pair_id").loc[ids].reset_index()


def take_sample(removed, texts=None, seed=0):
    if not len(removed):
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    picked = pd.concat([g.sample(min(len(g), SAMPLE_PER_REASON), random_state=seed)
                        for _, g in removed.groupby("reason")])
    if texts is None:
        texts = fetch_texts(active(), JOB["source"], picked["pair_id"])
    picked = picked.merge(texts[["pair_id", "kind", "query", "passage"]], on="pair_id", how="left")
    for col in ("query", "passage"):
        picked[col] = picked[col].str.slice(0, 600)
    return picked[SAMPLE_COLUMNS]


def finish_source(stage, step, source, alive_before, removed, samples, extra=None):
    d = stage.steps_dir / step
    d.mkdir(parents=True, exist_ok=True)
    alive = alive_before.copy()
    ids = removed["pair_id"].to_numpy(dtype=np.int64)
    if not alive[ids].all():
        raise AssertionError(f"{step}/{source}: a removed row was not alive")
    alive[ids] = False
    np.save(d / f"{source}.alive.npy", alive)
    write_parquet(removed[["pair_id", "reason", "detail"]].reset_index(drop=True), d / f"{source}.removed.parquet")
    kept = [s for s in samples if len(s)]
    sample = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=SAMPLE_COLUMNS)
    if len(sample):
        sample = pd.concat([g.sample(min(len(g), SAMPLE_PER_REASON), random_state=0)
                            for _, g in sample.groupby("reason")])
    write_parquet(sample.reset_index(drop=True), d / f"{source}.samples.parquet")
    return {"in": int(alive_before.sum()), "out": int(alive.sum()),
            "removed": {k: int(v) for k, v in removed["reason"].value_counts().items()}, **(extra or {})}


def save_step(stage, step, per_source, seconds):
    workers = 1 if (stage.smoke or not fork_available()) else stage.cfg.data.workers
    stats = {"step": step, "fingerprint": step_fingerprint(stage, step), "sources": per_source, "seconds": seconds,
             "workers": workers}
    (stage.steps_dir / step).mkdir(parents=True, exist_ok=True)
    (stage.steps_dir / step / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
    return stats


def load_samples(stage, step):
    parts = [pd.read_parquet(stage.steps_dir / step / f"{s}.samples.parquet").assign(source=s)
             for s in stage.source_names]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*SAMPLE_COLUMNS, "source"])


def format_removed(df, n, seed=1, detail_width=140):
    groups = [g.sample(frac=1, random_state=seed) for _, g in df.drop_duplicates(["query", "passage"]).groupby("reason")]
    picked, i = [], 0
    while len(picked) < n and any(i < len(g) for g in groups):
        picked += [g.iloc[i] for g in groups if i < len(g)][: n - len(picked)]
        i += 1
    lines = []
    for r in pd.DataFrame(picked).itertuples():
        lines.append(f"  - [{r.source}/{r.kind}] {r.reason}" + (f"  ({short(r.detail, detail_width)})" if r.detail else ""))
        lines.append(f"      Q: {short(r.query, 130)}")
        lines.append(f"      P: {short(r.passage, 200)}")
    return "\n".join(lines)


def report_step(stage, step, stats, n_examples=5):
    rows = sum(s["in"] for s in stats["sources"].values())
    stage.timings[step] = {"seconds": stats["seconds"], "rows": rows, "workers": stats["workers"]}
    lines = [f"{step}:", f"{'source':<10}{'rows in':>11}{'removed':>10}{'rows out':>11}   removed by reason"]
    for source, s in stats["sources"].items():
        reasons = ", ".join(f"{k}: {v:,}" for k, v in s["removed"].items()) or "-"
        lines.append(f"{source:<10}{s['in']:>11,}{s['in'] - s['out']:>10,}{s['out']:>11,}   {reasons}")
    total_out = sum(s["out"] for s in stats["sources"].values())
    lines.append(f"{'total':<10}{rows:>11,}{rows - total_out:>10,}{total_out:>11,}   "
                 f"({(rows - total_out) / max(rows, 1):.1%} removed, {stats['seconds']:.0f}s)")
    samples = load_samples(stage, step)
    if len(samples):
        n_distinct = len(samples.drop_duplicates(["query", "passage"]))
        lines += [f"{min(n_examples, n_distinct)} removed examples:", format_removed(samples, n_examples)]
    else:
        lines.append("nothing removed")
    log("\n".join(lines))
    return stats


def filter_shard(path):
    df = pd.read_parquet(path, columns=list(dict.fromkeys(["pair_id", "kind", "query", "passage", *JOB["columns"]])))
    df = df[JOB["alive"][df["pair_id"].to_numpy()]].reset_index(drop=True)
    reasons, details = JOB["decide"](df)
    reasons = np.asarray(reasons, dtype=object)
    hit = np.flatnonzero(np.array([r is not None for r in reasons], dtype=bool))
    removed = pd.DataFrame({"pair_id": df["pair_id"].to_numpy()[hit], "reason": reasons[hit].astype(str),
                            "detail": np.asarray(details, dtype=object)[hit].astype(str)})
    return {"n_in": len(df), "removed": removed,
            "sample": take_sample(removed, df.iloc[hit], seed=int(df["pair_id"].iloc[0]) if len(df) else 0)}


def run_row_filter(stage, step, columns, decide, prepare=None):
    if (stats := step_done(stage, step)) is not None:
        return stats
    t0, per_source = time.time(), {}
    for source in stage.source_names:
        alive = load_alive(stage, previous_step(step), source)
        JOB.clear()
        JOB.update(source=source, columns=columns, decide=decide, alive=alive)
        prepared = prepare(stage, source) if prepare else {}
        JOB.update(prepared.get("job", {}))
        results = parallel_map(filter_shard, clean_shards(stage, source), stage.cfg.data.workers,
                               desc=f"{step} {source}")
        if sum(r["n_in"] for r in results) != alive.sum():
            raise AssertionError(f"{step}/{source}: row count mismatch")
        removed = pd.concat([r["removed"] for r in results], ignore_index=True)
        per_source[source] = finish_source(stage, step, source, alive, removed, [r["sample"] for r in results],
                                           prepared.get("info"))
    return save_step(stage, step, per_source, time.time() - t0)


TAG_RE = re.compile(r"<!--.*?-->|</?[A-Za-z][A-Za-z0-9]*(?:\s[^<>]{0,300})?/?>", re.S)
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
SPACES_RE = re.compile(r"[^\S\n]+")
NEWLINES_RE = re.compile(r"\s*\n\s*")


def clean_text(text):
    rules, t = [], text or ""
    if "<" in t or "&" in t:
        new = html.unescape(TAG_RE.sub(" ", t))
        if new != t:
            rules.append("html")
            t = new
    if (new := URL_RE.sub(" ", t)) != t:
        rules.append("url")
        t = new
    if "@" in t and (new := EMAIL_RE.sub(" ", t)) != t:
        rules.append("email")
        t = new
    new = NEWLINES_RE.sub("\n", SPACES_RE.sub(" ", t)).strip()
    if new != t.strip():
        rules.append("whitespace")
    return new, rules


def fit_to_length(stage, texts, prefix, n_prefix, max_len, block=100_000):
    out, counts, cut = [], [], []
    for a in range(0, len(texts), block):
        part = texts[a: a + block]
        enc = stage.tokenizer([prefix + t for t in part], add_special_tokens=True, return_offsets_mapping=True)
        for t, ids, offs in zip(part, enc["input_ids"], enc["offset_mapping"]):
            if len(ids) <= max_len:
                out.append(t)
                counts.append(len(ids) - 2 - n_prefix)
                cut.append(False)
                continue
            k = max_len - 1
            while k > 1 + n_prefix and not stage.is_word_start[ids[k]]:
                k -= 1
            if k <= 1 + n_prefix:
                k = max_len - 1
            out.append(t[: offs[k - 1][1] - len(prefix)].rstrip())
            counts.append(k - 1 - n_prefix)
            cut.append(True)
    return out, counts, cut


def change_window(before, after, width=45):
    i = next((k for k, (a, b) in enumerate(zip(before, after)) if a != b), min(len(before), len(after)))
    lo = max(0, i - width)
    return repr(before[lo: i + width]), repr(after[lo: i + width])


def cleanup_shard(path):
    stage = active()
    source = path.parent.name
    df = pd.read_parquet(path)
    changes, examples = Counter(), {}
    for col, prefix, n_prefix, max_len in (("query", stage.q_prefix, stage.n_q_prefix, stage.max_q),
                                           ("passage", stage.p_prefix, stage.n_p_prefix, stage.max_p)):
        cleaned = []
        for text in df[col]:
            new, rules = clean_text(text)
            for rule in rules:
                changes[f"{col}: {rule}"] += 1
                if f"{col}: {rule}" not in examples and len(rules) == 1:
                    examples[f"{col}: {rule}"] = change_window(text, new)
            cleaned.append(new)
        fitted, n_tokens, cut = fit_to_length(stage, cleaned, prefix, n_prefix, max_len)
        if any(cut):
            i = cut.index(True)
            changes[f"{col}: cut to {max_len} tokens"] += int(sum(cut))
            examples.setdefault(f"{col}: cut to {max_len} tokens",
                                (f"{len(cleaned[i]):,} characters",
                                 f"{len(fitted[i]):,} characters, ends '…{fitted[i][-60:]}'"))
        df[col] = fitted
        df[f"{col[0]}_tokens"] = np.asarray(n_tokens, dtype=np.int32)
        df[f"{col[0]}_norm"] = [stage.normalize(t) for t in fitted]
    write_parquet(df, stage.clean / source / path.name)
    empty = ((df["q_tokens"] <= 0) | (df["p_tokens"] <= 0)).to_numpy()
    removed = pd.DataFrame({"pair_id": df.loc[empty, "pair_id"].to_numpy(), "reason": "empty after cleanup",
                            "detail": np.where(df.loc[empty, "q_tokens"] <= 0, "query", "passage")})
    return {"n_in": len(df), "changes": changes, "examples": examples, "removed": removed,
            "sample": take_sample(removed, df[empty])}


def run_cleanup(stage):
    if (stats := step_done(stage, "cleanup")) is not None:
        return stats
    t0, per_source = time.time(), {}
    for source in stage.source_names:
        (stage.clean / source).mkdir(parents=True, exist_ok=True)
        JOB.clear()
        JOB["source"] = source
        results = parallel_map(cleanup_shard, raw_shards(stage, source), stage.cfg.data.workers,
                               desc=f"cleanup {source}")
        changes, examples = Counter(), {}
        for r in results:
            changes.update(r["changes"])
            for k, v in r["examples"].items():
                examples.setdefault(k, v)
        removed = pd.concat([r["removed"] for r in results], ignore_index=True)
        per_source[source] = finish_source(stage, "cleanup", source, load_alive(stage, None, source), removed,
                                           [r["sample"] for r in results],
                                           {"changed": dict(changes), "examples": examples})
    return save_step(stage, "cleanup", per_source, time.time() - t0)


def format_cleanup_changes(stats):
    lines = ["rows changed (a row can be changed by several rules):"]
    for source, s in stats["sources"].items():
        lines.append(f"  {source:<10}" + (", ".join(f"{k}: {v:,}" for k, v in sorted(s["changed"].items())) or "none"))
    lines.append("one example per rule (the text around the first change, before -> after):")
    seen = set()
    for source, s in stats["sources"].items():
        for rule, (before, after) in s["examples"].items():
            if rule not in seen:
                seen.add(rule)
                lines.append(f"  [{source}] {rule}\n      before: {before}\n      after:  {after}")
    return "\n".join(lines)


def decide_length(df):
    f = active().cfg.filters
    reasons, details = [], []
    for q, p in zip(df["q_tokens"], df["p_tokens"]):
        if q < f.min_query_tokens:
            reasons.append(f"query < {f.min_query_tokens} tokens")
            details.append(f"{q} tokens")
        elif p < f.min_passage_tokens:
            reasons.append(f"passage < {f.min_passage_tokens} tokens")
            details.append(f"{p} tokens")
        else:
            reasons.append(None)
            details.append("")
    return reasons, details


ARABIC_LETTER = regex.compile(r"[\p{L}&&\p{Script=Arabic}]", flags=regex.V1)
ANY_LETTER = regex.compile(r"\p{L}")


def arabic_share(text):
    n = len(ANY_LETTER.findall(text))
    return len(ARABIC_LETTER.findall(text)) / n if n else 0.0


def decide_language(df):
    reasons, details, lo = [], [], active().cfg.filters.min_arabic_letters
    for q, p in zip(df["q_norm"], df["p_norm"]):
        if (sq := arabic_share(q)) < lo:
            reasons.append(f"query < {lo:.0%} Arabic letters")
            details.append(f"{sq:.0%}")
        elif (sp := arabic_share(p)) < lo:
            reasons.append(f"passage < {lo:.0%} Arabic letters")
            details.append(f"{sp:.0%}")
        else:
            reasons.append(None)
            details.append("")
    return reasons, details


PUNCT_RE = regex.compile(r"[\p{P}\p{S}]+")


def query_key(text_norm):
    return " ".join(PUNCT_RE.sub(" ", text_norm).lower().split())


def boilerplate_keys(stage):
    if not hasattr(stage, "boilerplate_keys"):
        stage.boilerplate_keys = {query_key(stage.normalize(t)) for t in stage.cfg.filters.boilerplate_titles}
    return stage.boilerplate_keys


def query_hashes(path):
    df = pd.read_parquet(path, columns=["pair_id", "q_norm"])
    df = df[JOB["alive"][df["pair_id"].to_numpy()]]
    return np.array([stable_hash(query_key(q)) for q in df["q_norm"]], dtype=np.int64)


def prepare_junk(stage, source):
    hashes = np.concatenate([np.zeros(0, np.int64), *parallel_map(query_hashes, clean_shards(stage, source),
                                                                  stage.cfg.data.workers)])
    keys, counts = np.unique(hashes, return_counts=True)
    many = counts > stage.cfg.filters.max_query_repeats
    boilerplate_keys(stage)
    return {"job": {"repeated": dict(zip(keys[many].tolist(), counts[many].tolist()))},
            "info": {"template queries": int(many.sum())}}


def decide_junk(df):
    stage = active()
    reasons, details, repeated = [], [], JOB["repeated"]
    boilerplate, limit = boilerplate_keys(stage), stage.cfg.filters.max_query_repeats
    for q in df["q_norm"]:
        key = query_key(q)
        if key in boilerplate:
            reasons.append("boilerplate title")
            details.append(key)
        elif (count := repeated.get(stable_hash(key))) is not None:
            reasons.append(f"query text > {limit} times in the source")
            details.append(f"{count}x: {short(key, 60)}")
        else:
            reasons.append(None)
            details.append("")
    return reasons, details


WORD_RE = re.compile(r"\w+")


def word_jaccard(a, b):
    x, y = set(WORD_RE.findall(a)), set(WORD_RE.findall(b))
    return len(x & y) / len(x | y) if x | y else 1.0


def decide_near_identical(df):
    limit = active().cfg.filters.max_jaccard
    reasons, details = [], []
    for q, p in zip(df["q_norm"], df["p_norm"]):
        if (j := word_jaccard(q, p)) > limit:
            reasons.append(f"query ≈ passage (word Jaccard > {limit})")
            details.append(f"Jaccard {j:.2f}")
        else:
            reasons.append(None)
            details.append("")
    return reasons, details


def run_basic_filters(stage):
    report_step(stage, "length", run_row_filter(stage, "length", ["q_tokens", "p_tokens"], decide_length))
    report_step(stage, "language", run_row_filter(stage, "language", ["q_norm", "p_norm"], decide_language))
    report_step(stage, "junk", run_row_filter(stage, "junk", ["q_norm"], decide_junk, prepare=prepare_junk))
    report_step(stage, "near_identical", run_row_filter(stage, "near_identical", ["q_norm", "p_norm"],
                                                        decide_near_identical))
