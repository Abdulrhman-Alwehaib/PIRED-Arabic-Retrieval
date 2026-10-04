import json
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import regex

from ..common.io import read_json, write_json, write_parquet
from ..common.runlog import log
from ..common.text import short
from ..contrastive.sources import answer_window

QUERY_COLUMNS = ["qid", "source", "domain", "query", "pos_pids", "neg_pids", "group", "qtype", "is_human"]


def wiki_passage(title, text):
    title, text = (title or "").strip(), (text or "").strip()
    return f"{title} {text}".strip() if title else text


def pick(stage, items, n, salt):
    items = sorted(items)
    if n is None or n >= len(items):
        return items
    return sorted(random.Random(f"{stage.cfg.data.seed}-{salt}").sample(items, n))


class SourceBuilder:
    def __init__(self, stage, source):
        self.stage, self.source = stage, source
        self.queries, self.passages, self.stats, self.t0 = [], {}, Counter(), time.time()

    def passage(self, domain, text, orig_id=""):
        pid = f"{domain}:{self.stage.text_key(text)}"
        if pid not in self.passages:
            self.passages[pid] = (domain, text, orig_id)
        return pid

    def add(self, orig_qid, query, pos_pids, neg_pids=(), group=None, qtype="", domain=None):
        s = self.stage.sources[self.source]
        pos = list(dict.fromkeys(pos_pids))
        self.queries.append({"qid": f"{self.source}:{orig_qid}", "source": self.source, "domain": domain or s.domain,
                             "query": " ".join(query.split()), "pos_pids": pos,
                             "neg_pids": [p for p in dict.fromkeys(neg_pids) if p not in pos],
                             "group": group or pos[0], "qtype": qtype, "is_human": s.human})

    def save(self, loading, budget):
        raw1 = self.stage.raw1
        q = pd.DataFrame(self.queries, columns=QUERY_COLUMNS)
        p = pd.DataFrame([(pid, d, t, o) for pid, (d, t, o) in self.passages.items()],
                         columns=["pid", "domain", "text", "orig_id"])
        write_parquet(q, raw1 / f"{self.source}.queries.parquet")
        write_parquet(p, raw1 / f"{self.source}.passages.parquet")
        summary = {"source": self.source, "budget": budget, "smoke": self.stage.smoke, "queries": len(q),
                   "passages": len(p), "positives": int(q["pos_pids"].map(len).sum()),
                   "negatives": int(q["neg_pids"].map(len).sum()), "stats": dict(self.stats), "loading": loading,
                   "seconds": time.time() - self.t0}
        write_json(summary, raw1 / f"{self.source}.summary.json")
        return summary


def cached_source(stage, source, budget):
    f = stage.raw1 / f"{source}.summary.json"
    if f.exists():
        s = read_json(f)
        if s["budget"] == budget and s["smoke"] == stage.smoke:
            log(f"{source}: built earlier, loaded from {stage.raw1} (delete {source}.* there to rebuild)")
            return s
    return None


def load_raw(stage, source):
    return (pd.read_parquet(stage.raw1 / f"{source}.queries.parquet"),
            pd.read_parquet(stage.raw1 / f"{source}.passages.parquet"))


def report_source(stage, summary):
    source = summary["source"]
    stage.loading[source] = summary["loading"]
    q, p = load_raw(stage, source)
    lines = [f"{source}: {summary['queries']:,} queries, {summary['passages']:,} passages, "
             f"{summary['positives'] / max(summary['queries'], 1):.2f} positives per query"
             + (f", {summary['negatives']:,} judged non-relevant passages" if summary["negatives"] else "")
             + f" ({summary['seconds']:.0f}s)", f"  loaded from: {summary['loading']}"]
    if summary["stats"]:
        lines.append("  counts: " + ", ".join(f"{k} {v:,}" for k, v in summary["stats"].items()))
    texts = p.set_index("pid")["text"]
    for r in q.sample(min(3, len(q)), random_state=0).itertuples():
        lines.append(f"    Q: {short(r.query, 120)}")
        lines.append(f"    P: {short(texts[r.pos_pids[0]], 200)}")
    log("\n".join(lines))


def smoke_budget(stage):
    return stage.cfg.data.smoke_queries if stage.smoke else None


def build_miracl(stage):
    d, budget = stage.cfg.data, smoke_budget(stage)
    if (s := cached_source(stage, "miracl", budget)) is not None:
        return s
    b = SourceBuilder(stage, "miracl")
    topics = stage.read_topics(d.miracl_repo, f"{d.miracl_dir}/topics/topics.{d.miracl_dir}-train.tsv", "miracl")
    qrels = stage.read_qrels(d.miracl_repo, f"{d.miracl_dir}/qrels/qrels.{d.miracl_dir}-train.tsv", "miracl")
    with_pos = sorted(set(qrels.loc[qrels["rel"] > 0, "qid"]) & set(topics))
    b.stats.update({"train topics": len(topics), "with a relevant passage": len(with_pos)})
    chosen = set(pick(stage, with_pos, budget, "miracl"))
    qrels = qrels[qrels["qid"].isin(chosen)]
    needed = set(qrels["docid"])
    texts = {docid: wiki_passage(title, text) for docid, title, text in stage.iter_miracl_corpus(needed.__contains__)}
    b.stats["judged passages found in the corpus"] = len(texts)
    for qid, g in qrels.groupby("qid"):
        pos = [b.passage("wiki", texts[x], f"miracl:{x}") for x in g.loc[g["rel"] > 0, "docid"] if x in texts]
        neg = [b.passage("wiki", texts[x], f"miracl:{x}") for x in g.loc[g["rel"] <= 0, "docid"] if x in texts]
        if pos:
            b.add(qid, topics[qid], pos, neg)
    return b.save(f"{d.miracl_repo} {d.miracl_dir}/topics + qrels (train, TSV); texts from {d.miracl_corpus_repo} "
                  f"{d.miracl_corpus_dir}/docs-*.jsonl.gz", budget)


def build_mrtydi(stage):
    d, budget = stage.cfg.data, smoke_budget(stage)
    if (s := cached_source(stage, "mrtydi", budget)) is not None:
        return s
    b = SourceBuilder(stage, "mrtydi")
    with stage.open_lines(d.mrtydi_repo, f"{d.mrtydi_dir}/train.jsonl.gz", "mrtydi", gz=True) as fh:
        for line in fh:
            r = json.loads(line)
            b.stats["train queries"] += 1
            pos = [b.passage("wiki", wiki_passage(p.get("title"), p.get("text")), f"mrtydi:{p['docid']}")
                   for p in r.get("positive_passages") or [] if (p.get("text") or "").strip()]
            if not pos:
                b.stats["skipped: no positive text"] += 1
                continue
            b.add(r["query_id"], r["query"], pos)
            if budget and len(b.queries) >= budget:
                break
    return b.save(f"{d.mrtydi_repo} {d.mrtydi_dir}/train.jsonl.gz (read directly; the repo is a script dataset)", budget)


def build_tydiqa(stage):
    d, budget = stage.cfg.data, smoke_budget(stage)
    if (s := cached_source(stage, "tydiqa", budget)) is not None:
        return s
    b = SourceBuilder(stage, "tydiqa")
    df = pd.read_parquet(stage.fetch_file(d.tydiqa_repo, d.tydiqa_train, "tydiqa"))
    df = df[df["id"].str.startswith("arabic")]
    b.stats["Arabic train rows"] = len(df)
    df = df[df["id"].isin(pick(stage, df["id"], budget, "tydiqa"))]
    for r in df.itertuples():
        answers = r.answers
        if not len(answers["text"]):
            b.stats["skipped: no answer"] += 1
            continue
        start, text = int(answers["answer_start"][0]), answers["text"][0]
        window = answer_window(r.context, start, start + len(text), d.answer_window_chars)
        if window is None:
            b.stats["skipped: answer longer than a passage"] += 1
            continue
        b.add(r.id, r.question, [b.passage("wiki", wiki_passage(r.title, window), f"tydiqa:{r.id}")])
    return b.save(f"{d.tydiqa_repo} {d.tydiqa_train} (main-branch parquet; rows whose id starts with 'arabic')", budget)


def build_arabicaqa(stage):
    d, budget = stage.cfg.data, smoke_budget(stage)
    if (s := cached_source(stage, "arabicaqa", budget)) is not None:
        return s
    b = SourceBuilder(stage, "arabicaqa")
    with open(stage.fetch_file(d.arabicaqa_repo, d.arabicaqa_train, "arabicaqa"), encoding="utf-8") as f:
        train_ids = {str(json.loads(line)["question_id"]) for line in f if line.strip()}
    b.stats["Open-ArabicaQA train questions"] = len(train_ids)
    chosen = set(pick(stage, train_ids, budget, "arabicaqa"))
    mrc = json.loads(stage.fetch_file(d.arabicaqa_mrc_repo, d.arabicaqa_mrc_train, "arabicaqa").read_text(
        encoding="utf-8"))
    for article in mrc["data"]:
        title = article.get("title") or ""
        for par in article["paragraphs"]:
            context = par["context"]
            for qa in par["qas"]:
                if str(qa["id"]) not in chosen:
                    continue
                b.stats["found in MRC/train.json"] += 1
                answers = qa.get("answers") or []
                text = (answers[0].get("text") or "").strip() if answers else ""
                if qa.get("is_impossible") or not text:
                    b.stats["skipped: no answer"] += 1
                    continue
                start = context.find(text, max(0, answers[0]["answer_start"] - 20))
                start = start if start >= 0 else context.find(text)
                if start < 0:
                    b.stats["skipped: answer not in the article"] += 1
                    continue
                window = answer_window(context, start, start + len(text), d.answer_window_chars)
                if window is None:
                    b.stats["skipped: answer longer than a passage"] += 1
                    continue
                b.add(str(qa["id"]), qa["question"], [b.passage("wiki", wiki_passage(title, window),
                                                                 f"arabicaqa:{qa['id']}")])
    return b.save(f"{d.arabicaqa_repo} {d.arabicaqa_train} (train question ids) + {d.arabicaqa_mrc_repo} "
                  f"{d.arabicaqa_mrc_train} (answer positions in the article)", budget)


def build_mmarco(stage):
    d, s = stage.cfg.data, stage.sources["mmarco"]
    budget = d.smoke_queries if stage.smoke else s.raw
    if (cached := cached_source(stage, "mmarco", budget)) is not None:
        return cached
    b = SourceBuilder(stage, "mmarco")
    qrels = pd.read_csv(stage.fetch_file(d.msmarco_qrels_repo, d.msmarco_qrels_file, "msmarco-qrels"), sep="\t",
                        dtype=int)
    qrels.columns = ["qid", "pid", "score"]
    qrels = qrels[qrels["score"] > 0]
    with open(stage.fetch_file(d.mmarco_repo, d.mmarco_queries, "mmarco"), encoding="utf-8") as f:
        queries = {int(a): t for a, _, t in (line.rstrip("\n").partition("\t") for line in f) if a.isdigit()}
    b.stats.update({"train queries": len(queries), "train qrels": len(qrels)})
    texts = {}
    if stage.smoke:
        texts = dict(stage.read_mmarco_collection(max_bytes=d.mmarco_smoke_bytes))
        qrels = qrels[qrels["pid"].isin(texts)]
    qids = pick(stage, sorted(set(qrels["qid"]) & set(queries)), budget, "mmarco")
    qrels = qrels[qrels["qid"].isin(set(qids))]
    if not stage.smoke:
        needed = set(qrels["pid"])
        texts = dict(stage.read_mmarco_collection(needed.__contains__))
    for qid, g in qrels.groupby("qid"):
        pos = [b.passage("mmarco", texts[p], f"mmarco:{p}") for p in g["pid"] if p in texts and texts[p].strip()]
        if pos:
            b.add(str(qid), queries[qid], pos)
    return b.save(f"{d.mmarco_repo} {d.mmarco_queries} + {d.mmarco_collection} (Google Translate); positives from "
                  f"{d.msmarco_qrels_repo} {d.msmarco_qrels_file}", budget)


LABELED_BUILDERS = {"miracl": build_miracl, "mrtydi": build_mrtydi, "tydiqa": build_tydiqa,
                    "arabicaqa": build_arabicaqa, "mmarco": build_mmarco}

PROSE_END = (".", "!", "؟", "?", ":", "؛", "…", "»", '"', ")")
DISAMBIGUATION_RE = regex.compile(r"قد (?:يشير|تشير)|يمكن ان (?:يشير|تشير)|يمكن أن (?:يشير|تشير)|صفحة توضيح")


def eval_positive_articles(stage):
    d = stage.cfg.data
    frames = [stage.read_qrels(d.miracl_repo, f"{d.miracl_dir}/qrels/qrels.{d.miracl_dir}-dev.tsv", "miracl")]
    frames += [stage.read_qrels(d.mrtydi_repo, f"{d.mrtydi_dir}/ir-format-data/qrels.{s}.txt", "mrtydi")
               for s in ("dev", "test")]
    q = pd.concat(frames)
    return set(q.loc[q["rel"] > 0, "docid"].str.split("#").str[0])


def not_prose_reason(title, text):
    if "توضيح" in title or DISAMBIGUATION_RE.search(text[:300]):
        return "disambiguation page"
    if title.startswith("قائمة"):
        return "list page"
    if text.count("|") >= 3 or text.count("\t") >= 3:
        return "table"
    chars = len(text) - text.count(" ")
    if chars and sum(c.isdigit() for c in text) / chars > 0.25:
        return "table or figures (over 25% digits)"
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) >= 3 and sum(len(line.split()) <= 5 for line in lines) / len(lines) > 0.5:
        return "list (short lines)"
    separators = sum(text.count(c) for c in ("،", ",", "؛", ";", "•")) + text.count(" - ")
    if separators >= 8 and len(text.split()) / (separators + 1) < 3.0:
        return "list (short items between commas)"
    return None


def text_tokens(stage, texts):
    return np.array([len(x) for x in stage.tokenizer(texts, add_special_tokens=False)["input_ids"]], dtype=np.int32)


def select_passages(stage, df, n, domain, salt):
    g, st, removed = stage.cfg.gen, Counter({"candidates": len(df)}), []
    reasons = [not_prose_reason(t, x) for t, x in zip(df["title"], df["text"])]
    df = df.assign(reason=reasons)
    for r, part in df[df["reason"].notna()].groupby("reason"):
        st[f"removed: {r}"] = len(part)
        removed.append(part.head(3).assign(detail=""))
    df = df[df["reason"].isna()]
    rng = np.random.default_rng([stage.cfg.data.seed, sum(map(ord, salt))])
    df = df.iloc[rng.permutation(len(df))]
    df = df[df.groupby("article").cumcount() < g.max_per_article]
    st["after at most 2 per article"] = len(df)
    df = df.head(min(len(df), 3 * n + 1_000)).copy()
    df["tokens"] = text_tokens(stage, df["passage"].tolist())
    max_tokens = stage.max_p - 2 - stage.n_p_prefix
    bad = (df["tokens"] < g.min_passage_tokens) | (df["tokens"] > max_tokens)
    st[f"removed: not {g.min_passage_tokens}-{max_tokens} tokens"] = int(bad.sum())
    removed.append(df[bad].head(3).assign(reason="length", detail=df.loc[bad, "tokens"].head(3).astype(str) + " tokens"))
    df = df[~bad]
    df["pid"] = [f"{domain}:{stage.text_key(t)}" for t in df["passage"]]
    dup = df["pid"].duplicated()
    st["removed: duplicate text"] = int(dup.sum())
    df = df[~dup].head(n)
    st["chosen"] = len(df)
    return df.assign(domain=domain), st, pd.concat(removed, ignore_index=True) if removed else pd.DataFrame()


def build_synthetic_passages(stage):
    path = stage.syn / "passages.parquet"
    if path.exists():
        log(f"synthetic passages: chosen earlier, loaded from {path}")
        return pd.read_parquet(path)
    g = stage.cfg.gen
    n_wiki = round(g.n_passages * g.wiki_share)
    n_news = g.n_passages - n_wiki
    excluded = eval_positive_articles(stage)
    wiki = stage.miracl_corpus_frame()
    wiki = wiki.assign(passage=[wiki_passage(t, x) for t, x in zip(wiki["title"], wiki["text"])])
    before = len(wiki)
    wiki = wiki[~wiki["article"].isin(excluded)]
    log(f"Wikipedia: {before:,} MIRACL corpus passages; {before - len(wiki):,} removed: their article holds a positive "
        f"of a MIRACL dev or Mr.TyDi dev/test query ({len(excluded):,} such articles)")
    wiki_sel, st_w, rem_w = select_passages(stage, wiki, n_wiki, "wiki", "wiki")
    news, news_from = stage.news_leads(3 * n_news + 1_000)
    news_sel, st_n, rem_n = select_passages(stage, news, n_news, "news", "news")
    for name, st in (("Wikipedia", st_w), ("news", st_n)):
        log(f"{name}: " + ", ".join(f"{k} {v:,}" for k, v in st.items()))
    removed = pd.concat([rem_w, rem_n], ignore_index=True)
    if len(removed):
        log("removed examples:\n" + "\n".join(f"  - {r.reason} {r.detail}: {short(r.passage, 170)}"
                                              for r in removed.drop_duplicates("reason").itertuples()))
    chosen = pd.concat([wiki_sel, news_sel], ignore_index=True)[["pid", "domain", "article", "passage"]]
    chosen = chosen.rename(columns={"passage": "text"})
    chosen = chosen.iloc[np.random.default_rng(stage.cfg.data.seed).permutation(len(chosen))].reset_index(drop=True)
    write_parquet(chosen, path)
    write_json({"wiki": st_w, "news": st_n, "news_from": news_from, "excluded_articles": len(excluded)},
               stage.syn / "passages.stats.json")
    return chosen
