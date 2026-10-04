import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import regex
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from ..common.io import read_json, write_json, write_parquet, write_text
from ..common.runlog import log
from ..common.text import fmt_hours, md_table, short
from ..contrastive.steps import arabic_share, change_window, clean_text, fit_to_length, query_key
from .queries import QUERY_COLUMNS, load_raw

REPAIR_STEPS = ("cleanup", "language", "mmarco_check", "length", "dedupe")
LATIN_WORD = regex.compile(r"^[\p{Script=Latin}\d\W]*\p{Script=Latin}[\p{Script=Latin}\d\W]*$")


def queries_fingerprint(queries):
    h = hashlib.sha256()
    for q in queries.values():
        h.update("".join(f"{a}\t{b}\n" for a, b in zip(q["qid"], q["query"])).encode("utf-8"))
    return h.hexdigest()[:16]


def step_fingerprint(name, settings, before):
    return hashlib.sha256((before + name + json.dumps(settings, sort_keys=True, ensure_ascii=False, default=str)
                           ).encode("utf-8")).hexdigest()[:16]


def removed_frame(q, mask, reasons, details, passages):
    mask = np.asarray(mask, dtype=bool)
    sub = q[mask]
    first = [p[0] if len(p) else None for p in sub["pos_pids"]]
    return pd.DataFrame({"source": sub["source"].to_numpy(), "qid": sub["qid"].to_numpy(),
                         "query": sub["query"].to_numpy(),
                         "reason": np.asarray(reasons, dtype=object)[mask],
                         "detail": np.asarray(details, dtype=object)[mask],
                         "passage": [passages.at[p, "text"] if p in passages.index else "" for p in first]})


def interleave(df, n, seed=1):
    groups = [g.sample(frac=1, random_state=seed) for _, g in df.groupby("reason")]
    picked, i = [], 0
    while len(picked) < n and any(i < len(g) for g in groups):
        picked += [g.iloc[i] for g in groups if i < len(g)][: n - len(picked)]
        i += 1
    return pd.DataFrame(picked)


def drop_passages(queries, passages, bad, reason_all, detail_of=None):
    out, removed = {}, []
    for s, q in queries.items():
        pos = [[p for p in lst if p not in bad] for lst in q["pos_pids"]]
        neg = [[p for p in lst if p not in bad] for lst in q["neg_pids"]]
        empty = np.array([not p for p in pos], dtype=bool)
        details = [detail_of(lst) if detail_of else "" for lst in q["pos_pids"]]
        removed.append(removed_frame(q, empty, [reason_all] * len(q), details, passages))
        q = q.assign(pos_pids=pos, neg_pids=neg)
        out[s] = q[~empty].reset_index(drop=True)
    return out, passages.drop(index=[p for p in bad if p in passages.index]), pd.concat(removed, ignore_index=True)


def used_pids(queries):
    used = set()
    for q in queries.values():
        for lst in q["pos_pids"]:
            used.update(lst)
        for lst in q["neg_pids"]:
            used.update(lst)
    return used


def check_cleanup(stage, queries, passages):
    changes, examples = Counter(), {}
    new_p = passages.copy()
    cleaned = []
    for text in passages["text"]:
        new, rules = clean_text(text)
        for rule in rules:
            changes[f"passage: {rule}"] += 1
            examples.setdefault(f"passage: {rule}", change_window(text, new))
        cleaned.append(new)
    new_p["text"] = cleaned
    empty_p = set(new_p.index[new_p["text"].str.len() == 0])
    out, removed = {}, []
    for s, q in queries.items():
        qs = []
        for text in q["query"]:
            new, rules = clean_text(text)
            for rule in rules:
                changes[f"query: {rule}"] += 1
                examples.setdefault(f"query: {rule}", change_window(text, new))
            qs.append(new)
        q = q.assign(query=qs)
        empty = (q["query"].str.len() == 0).to_numpy()
        removed.append(removed_frame(q, empty, ["empty after cleanup"] * len(q), [""] * len(q), new_p))
        out[s] = q[~empty].reset_index(drop=True)
    out, new_p, rem2 = drop_passages(out, new_p, empty_p, "passage empty after cleanup")
    return out, new_p, pd.concat(removed + [rem2], ignore_index=True), {
        "rows changed": dict(changes), "one example per rule (before -> after)": examples}


def check_language(stage, queries, passages):
    lo = stage.cfg.filters.min_arabic_letters
    share_p = {pid: arabic_share(stage.normalize(t)) for pid, t in zip(passages.index, passages["text"])}
    bad_p = {pid for pid, v in share_p.items() if v < lo}
    out, removed = {}, []
    for s, q in queries.items():
        share = np.array([arabic_share(stage.normalize(t)) for t in q["query"]])
        bad = share < lo
        removed.append(removed_frame(q, bad, [f"query < {lo:.0%} Arabic letters"] * len(q),
                                     [f"{v:.0%} Arabic" for v in share], passages))
        out[s] = q[~bad].reset_index(drop=True)
    out, passages, rem2 = drop_passages(out, passages, bad_p, f"no positive with >= {lo:.0%} Arabic letters",
                                        lambda lst: f"{share_p.get(lst[0], 0):.0%} Arabic" if lst else "")
    return out, passages, pd.concat(removed + [rem2], ignore_index=True), {"passages below the limit": len(bad_p)}


def broken_translation(f, text, is_query):
    words = text.split()
    if len(words) < (f.min_query_words if is_query else f.mmarco_min_passage_words):
        return "too short"
    latin_run = run = 0
    for w in words:
        run = run + 1 if LATIN_WORD.match(w) else 0
        latin_run = max(latin_run, run)
    if latin_run >= f.mmarco_latin_run:
        return f"untranslated: {latin_run} Latin words in a row"
    rep = best = 1
    for a, b in zip(words, words[1:]):
        rep = rep + 1 if a == b else 1
        best = max(best, rep)
    if best >= f.mmarco_repeat_run:
        return f"broken: a word repeated {best} times in a row"
    if len(words) >= 10 and len(set(words)) / len(words) < f.mmarco_min_distinct:
        return f"broken: only {len(set(words)) / len(words):.0%} distinct words"
    return None


def check_mmarco(stage, queries, passages):
    f = stage.cfg.filters
    q = queries["mmarco"]
    reasons = [broken_translation(f, t, True) for t in q["query"]]
    bad = np.array([r is not None for r in reasons])
    removed = [removed_frame(q, bad, [f"query {r}" if r else "" for r in reasons], [""] * len(q), passages)]
    out = dict(queries)
    out["mmarco"] = q[~bad].reset_index(drop=True)
    mm_pids = {p for lst in out["mmarco"]["pos_pids"] for p in lst}
    why = {pid: broken_translation(f, passages.at[pid, "text"], False) for pid in mm_pids}
    bad_p = {pid for pid, r in why.items() if r}
    out2, passages, rem2 = drop_passages({"mmarco": out["mmarco"]}, passages, bad_p, "positive passage broken",
                                         lambda lst: why.get(lst[0]) or "" if lst else "")
    out["mmarco"] = out2["mmarco"]
    return out, passages, pd.concat(removed + [rem2], ignore_index=True), {
        "passages removed by reason": dict(Counter(r.split(":")[0] for r in why.values() if r))}


def check_length(stage, queries, passages):
    f = stage.cfg.filters
    texts, n_tokens, cut = fit_to_length(stage, passages["text"].tolist(), stage.p_prefix, stage.n_p_prefix, stage.max_p)
    passages = passages.assign(text=texts, tokens=n_tokens)
    human_pids = {p for s, q in queries.items() if stage.sources[s].human for lst in q["pos_pids"] for p in lst}
    short_p = {pid for pid, n in zip(passages.index, passages["tokens"])
               if n < (f.min_passage_tokens_human if pid in human_pids else f.min_passage_tokens)}
    out, removed = {}, []
    for s, q in queries.items():
        n_q = [len(x) for x in stage.tokenizer([stage.q_prefix + t for t in q["query"]])["input_ids"]]
        words = [len(t.split()) for t in q["query"]]
        reasons = [f"query > {stage.max_q} tokens" if n > stage.max_q else f"query < {f.min_query_words} words"
                   if w < f.min_query_words else None for n, w in zip(n_q, words)]
        bad = np.array([r is not None for r in reasons])
        removed.append(removed_frame(q, bad, reasons, [f"{n} tokens, {w} words" for n, w in zip(n_q, words)],
                                     passages))
        out[s] = q[~bad].reset_index(drop=True)
    tok = passages["tokens"]
    out, passages, rem2 = drop_passages(out, passages, short_p, "no positive long enough",
                                        lambda lst: f"{tok.get(lst[0], 0)} tokens" if lst else "")
    return out, passages, pd.concat(removed + [rem2], ignore_index=True), {
        f"passages cut to {stage.max_p} tokens": int(sum(cut)), "short passages dropped": len(short_p)}


def check_dedupe(stage, queries, passages):
    names = stage.source_names
    first, merged_pos = {}, 0
    pos = {s: [list(x) for x in queries[s]["pos_pids"]] for s in names}
    neg = {s: [list(x) for x in queries[s]["neg_pids"]] for s in names}
    dup = {s: np.zeros(len(queries[s]), dtype=bool) for s in names}
    details = {s: [""] * len(queries[s]) for s in names}
    for s in names:
        q = queries[s]
        for i, (text, qid) in enumerate(zip(q["query"], q["qid"])):
            key = query_key(stage.normalize(text))
            if key not in first:
                first[key] = (s, i, qid)
                continue
            src, j, kept_qid = first[key]
            before = len(pos[src][j])
            pos[src][j] += [p for p in pos[s][i] if p not in pos[src][j]]
            merged_pos += len(pos[src][j]) > before
            neg[src][j] = [p for p in dict.fromkeys(neg[src][j] + neg[s][i]) if p not in pos[src][j]]
            dup[s][i], details[s][i] = True, f"same query as {kept_qid}"
    out, removed = {}, []
    for s in names:
        q = queries[s].assign(pos_pids=pos[s], neg_pids=neg[s])
        removed.append(removed_frame(q, dup[s], ["duplicate query (positives merged into the first copy)"] * len(q),
                                     details[s], passages))
        out[s] = q[~dup[s]].reset_index(drop=True)
    return out, passages, pd.concat(removed, ignore_index=True), {"kept queries that gained positives": merged_pos}


def check_step(stage, name):
    f = stage.cfg.filters
    return {"cleanup": ("cleanup", check_cleanup, {"version": 2}),
            "language": ("language", check_language, {"min_arabic_letters": f.min_arabic_letters}),
            "mmarco_check": ("mmarco_check", check_mmarco, {k: getattr(f, k) for k in (
                "mmarco_latin_run", "mmarco_repeat_run", "mmarco_min_distinct", "mmarco_min_passage_words",
                "min_query_words")}),
            "length": ("length", check_length, {k: getattr(f, k) for k in (
                "max_query_tokens", "max_passage_tokens", "min_query_words", "min_passage_tokens",
                "min_passage_tokens_human")}),
            "dedupe": ("dedupe", check_dedupe, {"version": 1})}[name]


class Part1State:
    def __init__(self, stage, queries, passages, funnel=None, step_stats=None):
        self.stage, self.queries, self.passages = stage, queries, passages
        self.fp = queries_fingerprint(queries)
        self.funnel = dict(funnel) if funnel else {"loaded": {s: len(q) for s, q in queries.items()}}
        self.step_stats = dict(step_stats or {})

    @classmethod
    def from_sources(cls, stage):
        queries = {s: load_raw(stage, s)[0] for s in stage.source_names}
        passages = pd.concat([load_raw(stage, s)[1] for s in stage.source_names], ignore_index=True)
        return cls(stage, queries, passages.drop_duplicates("pid").set_index("pid"))

    def save_state(self, name, queries, passages, fp, stats):
        d = self.stage.steps1 / name
        for s, q in queries.items():
            write_parquet(q, d / f"{s}.queries.parquet")
        write_parquet(passages.reset_index(), d / "passages.parquet")
        write_json({"fingerprint": fp, **stats}, d / "stats.json")

    def load_state(self, name, fp):
        d = self.stage.steps1 / name
        if not (d / "stats.json").exists() or read_json(d / "stats.json")["fingerprint"] != fp:
            return None
        queries = {s: pd.read_parquet(d / f"{s}.queries.parquet") for s in self.stage.source_names}
        passages = pd.read_parquet(d / "passages.parquet").set_index("pid")
        log(f"{name}: done earlier, loaded from {d} (delete that folder to run it again)")
        return queries, passages, read_json(d / "stats.json")

    def run_check(self, name, fn, settings):
        stage, names = self.stage, self.stage.source_names
        fp = step_fingerprint(name, settings, self.fp)
        cached = self.load_state(name, fp)
        if cached is not None:
            new_q, new_p, stats = cached
            removed = pd.read_parquet(stage.steps1 / name / "removed.parquet")
        else:
            t0 = time.time()
            new_q, new_p, removed, extra = fn(stage, self.queries, self.passages)
            stats = {"in": {s: len(q) for s, q in self.queries.items()}, "out": {s: len(q) for s, q in new_q.items()},
                     "removed": {s: dict(Counter(removed.loc[removed["source"] == s, "reason"])) for s in names},
                     "seconds": time.time() - t0, **extra}
            self.save_state(name, new_q, new_p, fp, stats)
            write_parquet(removed.groupby("reason", group_keys=False).head(200).reset_index(drop=True),
                          stage.steps1 / name / "removed.parquet")
        log(f"{name}:\n" + format_check(names, stats, removed))
        self.queries, self.passages, self.fp = new_q, new_p, fp
        self.funnel[name] = dict(stats["out"])
        self.step_stats[name] = stats
        return stats


def format_check(names, stats, removed):
    lines = [f"{'source':<11}{'queries in':>12}{'removed':>10}{'out':>11}   removed by reason"]
    for s in names:
        r = ", ".join(f"{k}: {v:,}" for k, v in stats["removed"][s].items()) or "-"
        lines.append(f"{s:<11}{stats['in'][s]:>12,}{stats['in'][s] - stats['out'][s]:>10,}{stats['out'][s]:>11,}   {r}")
    tin, tout = sum(stats["in"].values()), sum(stats["out"].values())
    lines.append(f"{'total':<11}{tin:>12,}{tin - tout:>10,}{tout:>11,}   ({(tin - tout) / max(tin, 1):.1%} removed, "
                 f"{stats['seconds']:.0f}s)")
    for k, v in stats.items():
        if k in ("in", "out", "removed", "seconds", "fingerprint"):
            continue
        if isinstance(v, dict) and v and all(isinstance(x, (list, tuple)) for x in v.values()):
            lines.append(f"{k}:")
            for rule, (a, b) in v.items():
                lines += [f"  {rule}", f"      before: {a}", f"      after:  {b}"]
        else:
            lines.append(f"{k}: {v}")
    if len(removed):
        lines.append(f"{min(5, len(removed))} removed examples:")
        for r in interleave(removed, 5).itertuples():
            lines.append(f"  - [{r.source}] {r.reason}" + (f" ({short(r.detail, 100)})" if r.detail else ""))
            lines.append(f"      Q: {short(r.query, 130)}")
            if r.passage:
                lines.append(f"      P: {short(r.passage, 180)}")
    else:
        lines.append("nothing removed")
    return "\n".join(lines)


def mteb_eval_queries(stage):
    import mteb

    f = stage.cfg.filters
    out = {}
    for task in mteb.get_tasks(languages=["ara"], task_types=["Retrieval"]):
        m = task.metadata
        if m.name in f.decontam_mteb_skip:
            continue
        repo, rev = m.dataset["path"], m.dataset.get("revision")
        subsets = [s for s in (task.hf_subsets or ["default"]) if s in f.decontam_mteb_subsets or s.startswith("ara-")]
        files = stage.api.list_repo_files(repo, repo_type="dataset", revision=rev)
        texts = []
        for subset in subsets:
            for split in m.eval_splits:
                wanted = [x for x in files if x.endswith((".parquet", ".jsonl", ".csv")) and (
                    x.startswith(f"{subset}-queries/{split}") or x.startswith(f"{split}/queries")
                    or (subset == "default" and x.startswith(f"queries/{split}")))]
                for x in wanted:
                    path = hf_hub_download(repo, x, repo_type="dataset", revision=rev, token=stage.token)
                    df = pd.read_parquet(path) if x.endswith(".parquet") else pd.read_json(path, lines=True)
                    col = next(c for c in ("text", "query", "question") if c in df.columns)
                    texts += df[col].astype(str).tolist()
        if not texts and any(x.endswith("arabic.csv") for x in files):
            x = next(x for x in files if x.endswith("arabic.csv"))
            texts = pd.read_csv(hf_hub_download(repo, x, repo_type="dataset", revision=rev,
                                                token=stage.token))["question"].astype(str).tolist()
        out[f"mteb {m.name}"] = texts
        if not texts:
            log(f"  WARNING: no query file found for {m.name} ({repo}); its queries are not decontaminated")
    return out


def eval_query_sets(stage):
    d = stage.cfg.data
    sets = {}
    for split in ("dev", "test-a", "test-b"):
        sets[f"MIRACL-ar {split}"] = list(stage.read_topics(
            d.miracl_repo, f"{d.miracl_dir}/topics/topics.{d.miracl_dir}-{split}.tsv", "miracl").values())
    for split in ("dev", "test"):
        sets[f"Mr.TyDi-ar {split}"] = list(stage.read_topics(
            d.mrtydi_repo, f"{d.mrtydi_dir}/ir-format-data/topics.{split}.txt", "mrtydi").values())
    dev = pd.read_parquet(stage.fetch_file(d.tydiqa_repo, d.tydiqa_dev, "tydiqa"))
    sets["TyDi QA Arabic dev"] = dev.loc[dev["id"].str.startswith("arabic"), "question"].tolist()
    for f in d.arabicaqa_eval:
        with open(stage.fetch_file(d.arabicaqa_repo, f, "arabicaqa"), encoding="utf-8") as fh:
            sets[f"Open-ArabicaQA {Path(f).stem}"] = [json.loads(line)["question"] for line in fh if line.strip()]
    sets.update(mteb_eval_queries(stage))
    return {k: [t for t in v if isinstance(t, str) and t.strip()] for k, v in sets.items()}


class Decontaminator:
    def __init__(self, stage):
        f = stage.cfg.filters
        self.stage, self.ngram = stage, f.decontam_ngram
        self.sets = eval_query_sets(stage)
        self.exact, self.ngrams = {}, defaultdict(dict)
        for name, texts in self.sets.items():
            for text in texts:
                key = query_key(stage.normalize(text))
                words, label = key.split(), f"{name}: {short(text, 70)}"
                self.exact.setdefault(key, label)
                if len(words) >= f.decontam_ngram:
                    for i in range(len(words) - f.decontam_ngram + 1):
                        self.ngrams[f.decontam_ngram].setdefault(tuple(words[i: i + f.decontam_ngram]), label)
                elif len(words) >= f.decontam_min_words:
                    self.ngrams[len(words)].setdefault(tuple(words), label)
        log("evaluation queries: " + ", ".join(f"{k} {len(v):,}" for k, v in self.sets.items())
            + f"; {len(self.exact):,} distinct; n-gram keys by length: "
            + str({n: len(g) for n, g in sorted(self.ngrams.items())}))
        dev_query = next(t for t in self.sets["MIRACL-ar dev"] if 6 <= len(t.split()) <= 12)
        self.dev_query = dev_query
        self.test_queries = {"miracl:decontam-test-exact": "  " + dev_query + " ",
                             "miracl:decontam-test-inside": "سؤال: " + dev_query + " وما أهميته؟"}

    def hit(self, query):
        key = query_key(self.stage.normalize(query))
        if (label := self.exact.get(key)) is not None:
            return "matches an evaluation query", label
        words = key.split()
        for n, grams in self.ngrams.items():
            for i in range(len(words) - n + 1):
                if (label := grams.get(tuple(words[i: i + n]))) is not None:
                    return ("shares a 13-gram with an evaluation query" if n == self.ngram
                            else "contains a whole evaluation query (6-12 words)"), label
        return None

    def check(self, stage, queries, passages):
        queries = dict(queries)
        m = queries["miracl"]
        fake = pd.DataFrame([{**m.iloc[0].to_dict(), "qid": qid, "query": text}
                             for qid, text in self.test_queries.items()])
        queries["miracl"] = pd.concat([m, fake], ignore_index=True)
        out, removed = {}, []
        for s, q in queries.items():
            hits = [self.hit(t) for t in q["query"]]
            bad = np.array([h is not None for h in hits])
            removed.append(removed_frame(q, bad, [h[0] if h else None for h in hits],
                                         [h[1] if h else "" for h in hits], passages))
            out[s] = q[~bad].reset_index(drop=True)
        removed = pd.concat(removed, ignore_index=True)
        test_rows = removed[removed["qid"].isin(self.test_queries)]
        removed = removed[~removed["qid"].isin(self.test_queries)]
        return out, passages, removed, {"inserted test queries removed": test_rows[["qid", "reason"]].values.tolist()}

    def settings(self):
        f = self.stage.cfg.filters
        return {"sets": {k: len(v) for k, v in self.sets.items()}, "ngram": f.decontam_ngram,
                "min_words": f.decontam_min_words, "test": self.test_queries}

    def run(self, state):
        state.run_check("decontam", self.check, self.settings())
        test = state.step_stats["decontam"]["inserted test queries removed"]
        if len(test) != len(self.test_queries) or any(state.queries["miracl"]["qid"].isin(self.test_queries)):
            raise AssertionError(f"the inserted evaluation queries were not all removed: {test}")
        result = (f"PASS the MIRACL dev query {self.dev_query!r} inserted as a training query "
                  f"({len(self.test_queries)} forms: exact with extra spaces, and inside a longer query) was removed: "
                  + "; ".join(f"{q} -> {r}" for q, r in test))
        log(result)
        return result


def holdout(q, n, seed):
    n = min(n, len(q) // 5)
    groups = q["group"].drop_duplicates().sample(frac=1, random_state=seed).tolist()
    sizes = q["group"].value_counts()
    chosen, total = set(), 0
    for g in groups:
        if total >= n:
            break
        chosen.add(g)
        total += int(sizes[g])
    return q["group"].isin(chosen).to_numpy()


def save_part1(state, stats):
    stage = state.stage
    f, final1 = stage.cfg.filters, stage.final1
    parts = []
    for i, s in enumerate(stage.source_names):
        q = state.queries[s].copy()
        n_val = min(f.val_queries_per_source, f.val_queries_max.get(s, f.val_queries_per_source))
        q["split"] = np.where(holdout(q, n_val, stage.cfg.data.seed + i), "validation", "train")
        parts.append(q)
    queries = pd.concat(parts, ignore_index=True)
    used = used_pids({"all": queries})
    passages = state.passages.loc[sorted(used)].reset_index()[["pid", "domain", "text", "orig_id"]]
    shutil.rmtree(final1, ignore_errors=True)
    write_parquet(queries, final1 / "queries.parquet")
    write_parquet(passages, final1 / "passages.parquet")
    counts = {s: {"train": int(((queries["source"] == s) & (queries["split"] == "train")).sum()),
                  "validation": int(((queries["source"] == s) & (queries["split"] == "validation")).sum())}
              for s in stage.source_names}
    manifest = {"part": 1, "smoke": stage.smoke, "queries": counts, "passages": len(passages),
                "columns": QUERY_COLUMNS + ["split"], "prefixes": {"query": stage.q_prefix, "passage": stage.p_prefix},
                "max_tokens": {"query": stage.max_q, "passage": stage.max_p},
                "tokenizer_fingerprint": stage.tokenizer_fingerprint, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json(manifest, final1 / "manifest.json")
    write_json({"funnel": state.funnel, "steps": state.step_stats, **stats}, final1 / "stats.json")
    return manifest


def data_report(stage, manifest, stats):
    names, g = stage.source_names, stage.cfg.gen
    funnel = stats["funnel"]
    steps = list(funnel)
    rows = [[s, *[funnel[st].get(s, 0) for st in steps], manifest["queries"][s]["train"],
             manifest["queries"][s]["validation"]] for s in names]
    rows.append(["total", *[sum(funnel[st].values()) for st in steps],
                 sum(v["train"] for v in manifest["queries"].values()),
                 sum(v["validation"] for v in manifest["queries"].values())])
    reasons = Counter()
    for st, st_stats in stats["steps"].items():
        for src in st_stats["removed"].values():
            for k, n in src.items():
                reasons[(st, k)] += n
    syn = stats.get("synthetic_passages") or {}
    gen = stats.get("generation") or {}
    return "\n\n".join([
        f"# Stage 4 data: supervised queries{' (SMOKE RUN: small subsets)' if stage.smoke else ''}",
        f"Built by `pired.supervised` (Part 1) on {time.strftime('%Y-%m-%d')}. Query records: "
        f"`{', '.join(QUERY_COLUMNS)}, split`; passages: `pid, domain, text, orig_id`. Prefixes at training time: "
        f"`{stage.q_prefix}` / `{stage.p_prefix}`; limits {stage.max_q} / {stage.max_p} tokens.",
        "## Sources",
        md_table(["source", "what", "loaded from", "license"],
                 [[s, stage.sources[s].description, (stats.get("loading") or {}).get(s, ""), stage.sources[s].license]
                  for s in names]),
        "The train splits of MIRACL and Mr.TyDi are used for training: MIRACL-ar and Mr.TyDi-ar are in-domain "
        "benchmarks from this stage on. Dev and test queries of every evaluated task were removed (decontamination).",
        "## Synthetic questions",
        md_table(["", "value"], [
            ["passages (Wikipedia / news)",
             f"{syn.get('wiki', {}).get('chosen', 0):,} / {syn.get('news', {}).get('chosen', 0):,}"],
            ["articles excluded (hold MIRACL dev / Mr.TyDi dev+test positives)", f"{syn.get('excluded_articles', 0):,}"],
            ["generator", g.smoke_model if stage.smoke else g.model],
            ["generation time", fmt_hours(gen["seconds"]) if gen.get("seconds") else "not recorded"],
            ["tokens per passage (prompt / output)",
             f"{gen.get('prompt_tokens', 0) / max(gen.get('passages', 1), 1):.0f} / "
             f"{gen.get('completion_tokens', 0) / max(gen.get('passages', 1), 1):.0f}"]]),
        "## Funnel (queries left after each step)",
        md_table(["source", *steps, "train", "validation"], rows),
        *([f"**Note:** {stats['note']}"] if stats.get("note") else []),
        "## Removed, by step and reason",
        md_table(["step", "reason", "queries"], [[st, k, n] for (st, k), n in reasons.items()]),
        f"Decontamination test: {stats.get('decontam_test')}",
    ])


def write_data_report(stage, manifest):
    report = data_report(stage, manifest, read_json(stage.final1 / "stats.json"))
    for path in (stage.cfg.reports_dir / "stage_supervised_data.md", stage.final1 / "README.md"):
        write_text(report, path)
    return report


def has_all_checks(stats):
    return all(s in (stats.get("steps") or {}) for s in REPAIR_STEPS)


def repair_part1(stage, folder):
    stats = read_json(folder / "stats.json")
    q = pd.read_parquet(folder / "queries.parquet").drop(columns="split")
    queries = {s: q[q["source"] == s].reset_index(drop=True) for s in stage.source_names}
    passages = pd.read_parquet(folder / "passages.parquet").set_index("pid")
    state = Part1State(stage, queries, passages, stats["funnel"], stats["steps"])
    for name in REPAIR_STEPS:
        state.run_check(*check_step(stage, name))
    note = (f"the checks {', '.join(REPAIR_STEPS)} ran after the decontamination (Part 2's repair step, "
            f"{time.strftime('%Y-%m-%d')}): the first upload of this data had skipped them. The validation split was "
            "drawn again with the same seeds.")
    keep = ("loading", "generation", "generation_test", "decontam_test", "synthetic_passages")
    return save_part1(state, {**{k: stats.get(k) for k in keep}, "note": note})


def hub_has_all_checks(stage, repo_id):
    try:
        path = hf_hub_download(repo_id, "stats.json", repo_type="dataset", token=stage.token,
                               cache_dir=stage.work / ".hub_check", force_download=True)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return False
    return has_all_checks(read_json(path))


def replace_on_hub(stage, folder, repo_id, keep="synthetic/"):
    api = stage.api
    files = {p.relative_to(folder).as_posix(): p for p in sorted(Path(folder).rglob("*"))
             if p.is_file() and ".cache" not in p.parts}
    size = sum(p.stat().st_size for p in files.values()) / 1e6
    if not stage.cfg.hub.push or not stage.token:
        log(f"[not uploaded] would replace the files of {repo_id} (except {keep}) with {len(files)} files "
            f"({size:,.1f} MB) from {folder}")
        return
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    stale = [f for f in api.list_repo_files(repo_id, repo_type="dataset")
             if f not in files and f != ".gitattributes" and not f.startswith(keep)]
    operations = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(p)) for k, p in files.items()]
    operations += [CommitOperationDelete(path_in_repo=f) for f in stale]
    api.create_commit(repo_id, operations=operations, repo_type="dataset",
                      commit_message="Part 1 output with the checks G1-G5 (repair)")
    log(f"{repo_id}: {len(files)} files uploaded ({size:,.1f} MB), {len(stale)} old files deleted, {keep} kept")


def ensure_checked_part1(stage):
    repo = stage.repo_id("queries")
    stage.pull_folder(repo, "dataset", stage.final1)
    if has_all_checks(read_json(stage.final1 / "stats.json")):
        log(f"Part 1's output in {stage.final1} has the checks G1-G5: nothing to repair")
    else:
        t0 = time.time()
        manifest = repair_part1(stage, stage.final1)
        report = write_data_report(stage, manifest)
        write_text(report, stage.final1 / "reports" / "stage_supervised_data.md")
        stage.announce(f"Part 1 repaired (checks G1-G5) in {fmt_hours(time.time() - t0)}: "
                       f"{sum(v['train'] + v['validation'] for v in manifest['queries'].values()):,} queries")
    if stage.cfg.hub.push and stage.token and hub_has_all_checks(stage, repo):
        log(f"{repo} holds a Part 1 output with the checks G1-G5")
    else:
        replace_on_hub(stage, stage.final1, repo)
