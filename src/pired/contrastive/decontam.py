import json
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

from ..common.runlog import log
from ..common.text import short
from .config import EVAL_QUERY_FILES
from .context import active
from .steps import JOB, query_key, run_row_filter


def load_eval_queries(token):
    out = {}
    for name, (repo, files) in EVAL_QUERY_FILES.items():
        rows = []
        for f in files:
            lines = Path(hf_hub_download(repo, f, repo_type="dataset", token=token)).read_text(
                encoding="utf-8").splitlines()
            if f.endswith(".jsonl"):
                rows += [(f, json.loads(line)["question"]) for line in lines if line.strip()]
            else:
                rows += [(f, line.split("\t", 1)[1]) for line in lines if "\t" in line]
        out[name] = rows
    return out


def decontam_tables(stage):
    if hasattr(stage, "decontam_tables"):
        return stage.decontam_tables
    f = stage.cfg.filters
    eval_queries = load_eval_queries(stage.token)
    exact, ngrams = {}, defaultdict(dict)
    for name, rows in eval_queries.items():
        for file, text in rows:
            key = query_key(stage.normalize(text))
            words, label = key.split(), f"{name} {Path(file).name}: {short(text, 70)}"
            exact.setdefault(key, label)
            if len(words) >= f.decontam_ngram:
                for i in range(len(words) - f.decontam_ngram + 1):
                    ngrams[f.decontam_ngram].setdefault(tuple(words[i: i + f.decontam_ngram]), label)
            elif len(words) >= f.decontam_min_words:
                ngrams[len(words)].setdefault(tuple(words), label)
    log("evaluation queries: " + ", ".join(f"{k} {len(v):,}" for k, v in eval_queries.items())
        + f"; {len(exact):,} distinct, n-gram keys by length: " + str({n: len(g) for n, g in sorted(ngrams.items())}))
    stage.decontam_tables = (exact, dict(ngrams))
    return stage.decontam_tables


def prepare_decontam(stage, source):
    exact, ngrams = decontam_tables(stage)
    return {"job": {"exact": exact, "ngrams": ngrams}}


def decide_decontam(df):
    long_ngram = active().cfg.filters.decontam_ngram
    exact, ngrams = JOB["exact"], JOB["ngrams"]
    reasons, details = [], []
    for q in df["q_norm"]:
        key = query_key(q)
        if (label := exact.get(key)) is not None:
            reasons.append("matches an evaluation query")
            details.append(label)
            continue
        words, hit = key.split(), None
        for n, grams in ngrams.items():
            for i in range(len(words) - n + 1):
                if (label := grams.get(tuple(words[i: i + n]))) is not None:
                    hit = ("shares a 13-gram with an evaluation query" if n == long_ngram
                           else "contains a whole evaluation query (6-12 words)", label)
                    break
            if hit:
                break
        reasons.append(hit[0] if hit else None)
        details.append(hit[1] if hit else "")
    return reasons, details


def run_decontam(stage):
    return run_row_filter(stage, "decontam", ["q_norm"], decide_decontam, prepare=prepare_decontam)
