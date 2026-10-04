import hashlib
import json
import math
import random
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..common.io import write_json, write_parquet
from ..common.parallel import parallel_map
from ..common.runlog import log
from ..common.text import short
from .context import active


@dataclass
class Unit:
    source: str
    uid: str
    maker: str
    repo: str = ""
    path: str = ""
    local_name: str = ""
    revision: str | None = None
    row_groups: tuple[int, int] | None = None
    limit: int | None = None


def plan_row_group_units(stage, source, maker, repo, files, local_name, per_unit, revision=None):
    rng = random.Random(f"{stage.cfg.data.seed}-{source}")
    files = list(files)
    rng.shuffle(files)
    per = 1 if stage.smoke else per_unit
    units = []
    for path in files:
        n = stage.num_row_groups(repo, path, local_name, revision)
        units += [Unit(source, f"{Path(path).stem}-rg{s:05d}", maker, repo, path, local_name, revision,
                       (s, min(s + per, n))) for s in range(0, n, per)]
    if stage.smoke:
        rng.shuffle(units)
    return units


def iter_row_groups(unit, columns):
    pf = active().open_parquet(unit.repo, unit.path, unit.local_name, unit.revision)
    start, end = unit.row_groups or (0, pf.metadata.num_row_groups)
    for rg in range(start, end):
        yield pf.read_row_group(rg, columns=columns).to_pylist()


def unit_file(stage, source, uid):
    return stage.raw / source / "units" / f"{uid}.parquet"


def build_unit(unit):
    stage = active()
    meta = unit_file(stage, unit.source, unit.uid).with_suffix(".json")
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    t0 = time.time()
    pairs, stats = MAKERS[unit.maker](unit)
    write_parquet(pd.DataFrame(pairs, columns=["kind", "query", "passage"]), unit_file(stage, unit.source, unit.uid))
    result = {"uid": unit.uid, "pairs": len(pairs), "seconds": time.time() - t0, "stats": stats}
    meta.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def write_shuffled_shards(stage, name, parts, n_rows):
    out_dir, bucket_dir = stage.raw / name, stage.raw / name / "buckets"
    for old in out_dir.glob("shard-*.parquet"):
        old.unlink()
    shutil.rmtree(bucket_dir, ignore_errors=True)
    bucket_dir.mkdir(parents=True)
    n_shards = max(1, math.ceil(n_rows / stage.cfg.data.shard_rows))
    rng = np.random.default_rng([stage.cfg.data.seed, stage.source_names.index(name), 1])
    writers, kinds = {}, Counter()
    for df in parts:
        kinds.update(df["kind"].tolist())
        for bucket, part in df.groupby(rng.integers(n_shards, size=len(df)), sort=False):
            table = pa.Table.from_pandas(part.reset_index(drop=True), preserve_index=False)
            if bucket not in writers:
                writers[bucket] = pq.ParquetWriter(bucket_dir / f"{bucket:05d}.parquet", table.schema)
            writers[bucket].write_table(table)
    for writer in writers.values():
        writer.close()
    starts, next_id = [], 0
    for k, bucket in enumerate(sorted(writers)):
        df = pd.read_parquet(bucket_dir / f"{bucket:05d}.parquet")
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
        df.insert(0, "pair_id", np.arange(next_id, next_id + len(df), dtype=np.int64))
        write_parquet(df, out_dir / f"shard-{k:05d}.parquet")
        starts.append(next_id)
        next_id += len(df)
    shutil.rmtree(bucket_dir, ignore_errors=True)
    if next_id != n_rows:
        raise AssertionError(f"{name}: wrote {next_id} rows, expected {n_rows}")
    return starts, kinds


def build_source(stage, name, units):
    cfg = stage.cfg
    budget = cfg.data.smoke_rows if stage.smoke else stage.sources[name].raw
    units_key = hashlib.sha256(json.dumps([[u.repo, u.path, u.row_groups, u.maker, u.limit] for u in units]).encode()
                               ).hexdigest()[:16]
    summary_file = stage.raw / name / "summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary["budget"] == budget and summary.get("units_key") == units_key:
            log(f"{name}: already built, {summary['pairs']:,} pairs (delete {stage.raw / name} to rebuild)")
            return summary
    t0, results = time.time(), []
    wave = 1 if stage.smoke else max(1, cfg.data.workers)
    for i in range(0, len(units), wave):
        batch = units[i: i + wave]
        if not stage.smoke:
            files = list(dict.fromkeys((u.repo, u.path, u.local_name, u.revision) for u in batch
                                       if u.path and not unit_file(stage, u.source, u.uid).with_suffix(".json").exists()))
            with ThreadPoolExecutor(8) as pool:
                list(pool.map(lambda f: stage.fetch_file(*f), files))
        results += parallel_map(build_unit, batch, cfg.data.workers, desc=f"{name} units {i}-{i + wave}")
        made = sum(r["pairs"] for r in results)
        if not stage.smoke:
            log(f"{name}: {len(results)}/{len(units)} units, {made:,} pairs")
        if budget and made >= budget:
            break
    counts = [r["pairs"] for r in results]
    made = sum(counts)
    keep = None
    if budget and made > budget:
        keep = np.zeros(made, dtype=bool)
        keep[np.random.default_rng([cfg.data.seed, stage.source_names.index(name)]).choice(made, budget,
                                                                                           replace=False)] = True
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(int)

    def parts():
        for r, a, b in zip(results, offsets[:-1], offsets[1:]):
            df = pd.read_parquet(unit_file(stage, name, r["uid"]))
            yield df if keep is None else df[keep[a:b]]

    n_rows = int(keep.sum()) if keep is not None else made
    starts, kinds = write_shuffled_shards(stage, name, parts(), n_rows)
    stats = Counter()
    for r in results:
        stats.update(r["stats"])
    summary = {"source": name, "budget": budget, "units_key": units_key, "units_done": len(results),
               "units_total": len(units), "made": made, "pairs": n_rows, "shard_starts": starts, "kinds": dict(kinds),
               "stats": dict(stats), "seconds": time.time() - t0, "unit_seconds": sum(r["seconds"] for r in results)}
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def raw_shards(stage, name):
    return sorted((stage.raw / name).glob("shard-*.parquet"))


def describe_pairs(df, n, seed=0, width=220):
    lines = []
    for r in df.sample(min(n, len(df)), random_state=seed).itertuples():
        lines.append(f"    [{r.kind}] Q: {short(r.query, 130)}")
        lines.append(f"    {' ' * (len(r.kind) + 2)} P: {short(r.passage, width)}")
    return "\n".join(lines)


def report_source(stage, name, summary, loading):
    stage.loading[name] = loading
    budget = summary["budget"]
    stage.timings[f"build_{name}"] = {"seconds": summary["unit_seconds"], "rows": summary["made"], "workers": 1}
    log("\n".join([
        f"{name}: {summary['pairs']:,} raw pairs kept (budget {f'{budget:,}' if budget else 'all'}; made "
        f"{summary['made']:,} in {summary['units_done']}/{summary['units_total']} units, {summary['seconds']:.0f}s)",
        f"  loaded from: {loading}",
        "  by kind: " + ", ".join(f"{k} {v:,}" for k, v in sorted(summary["kinds"].items())),
        "  counts: " + ", ".join(f"{k} {v:,}" for k, v in summary["stats"].items()),
        "  examples:",
        describe_pairs(pd.read_parquet(raw_shards(stage, name)[0]), 3),
    ]))


TITLE_END = tuple(".!?؟،,;؛:")


def is_heading(raw_line, prev_blank, next_blank):
    line = raw_line.strip()
    return (0 < len(line.split()) <= 10 and len(line) <= 80 and not line.endswith(TITLE_END)
            and (raw_line.endswith(" ") or (prev_blank and next_blank)))


def paragraphs(text):
    return [p.strip() for p in (text or "").split("\n") if p.strip()]


def join_until(paras, min_words, stop=None):
    out, words = [], 0
    for p in paras:
        if out and stop is not None and stop(p):
            break
        out.append(p)
        words += len(p.split())
        if words >= min_words:
            break
    return "\n".join(out)[: active().cfg.data.max_text_chars]


def make_news(unit):
    pairs, st = [], Counter()
    for rows in iter_row_groups(unit, ["head_line", "text"]):
        for r in rows:
            st["articles"] += 1
            head, paras = (r["head_line"] or "").strip(), paragraphs(r["text"])
            if not head or not paras:
                st["skipped: no headline or no text"] += 1
                continue
            pairs.append(("news", head, join_until(paras, active().cfg.data.news_min_words)))
        if unit.limit and len(pairs) >= unit.limit:
            break
    return pairs, dict(st)


WIKI_END_HEADINGS = (
    "مراجع", "المراجع", "مصادر", "المصادر", "انظر أيضا", "انظر أيضًا", "انظر أيضاً", "وصلات خارجية", "روابط خارجية",
    "هوامش", "الهوامش", "ملاحظات", "الملاحظات", "قراءات إضافية", "قراءات أخرى", "مراجع وهوامش", "المراجع والهوامش",
    "مصادر ومراجع", "المصادر والمراجع", "الملاحظات والمراجع", "المراجع والملاحظات", "ملاحظات ومراجع",
    "مصادر وحواشي", "حواشي", "وصلات", "ببليوغرافيا", "بيبليوغرافيا", "معرض الصور", "معرض صور")
PROSE_END = (".", "!", "؟", "?", ":", "؛", "…", "»", '"', ")")


def wiki_end_sections(stage):
    if not hasattr(stage, "wiki_end_sections"):
        stage.wiki_end_sections = {stage.normalize(h) for h in WIKI_END_HEADINGS}
    return stage.wiki_end_sections


def is_prose(line):
    return len(line.split()) >= 12 or line.endswith(PROSE_END)


def wiki_structure(text):
    stage = active()
    end_sections = wiki_end_sections(stage)
    lines = text.split("\n")
    lead, sections, heading, body = [], [], None, []
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        prev_blank = i == 0 or not lines[i - 1].strip()
        next_blank = i + 1 == len(lines) or not lines[i + 1].strip()
        if (lead or heading is not None) and is_heading(raw, prev_blank, next_blank):
            if heading is not None:
                sections.append((heading, body))
            if stage.normalize(line) in end_sections:
                return lead, sections
            heading, body = line, []
        elif heading is None:
            lead.append(line)
        else:
            body.append(line)
    if heading is not None:
        sections.append((heading, body))
    return lead, sections


def make_wiki(unit):
    d = active().cfg.data
    pairs, st, min_words = [], Counter(), d.wiki_min_words
    for rows in iter_row_groups(unit, ["title", "text"]):
        for r in rows:
            st["articles"] += 1
            title = (r["title"] or "").strip()
            if not title or "(توضيح)" in title:
                st["skipped: disambiguation page"] += 1
                continue
            lead, sections = wiki_structure(r["text"] or "")
            lead_text = "\n".join(line for line in lead if is_prose(line))
            if len(lead_text.split()) >= min_words:
                pairs.append(("wiki_lead", title, lead_text[: d.max_text_chars]))
            else:
                st["lead shorter than min_words"] += 1
            for heading, body in sections:
                text = "\n".join(line for line in body if is_prose(line))
                st["sections"] += 1
                if len(text.split()) >= min_words:
                    pairs.append(("wiki_section", f"{title} {heading}", text[: d.max_text_chars]))
                else:
                    st["section shorter than min_words"] += 1
        if unit.limit and len(pairs) >= unit.limit:
            break
    return pairs, dict(st)


def make_xlsum(unit):
    pairs, st = [], Counter()
    for rows in iter_row_groups(unit, ["title", "summary", "text"]):
        for r in rows:
            st["articles"] += 1
            title, summary, text = ((r[k] or "").strip() for k in ("title", "summary", "text"))
            if title and summary:
                pairs.append(("xlsum_title", title, summary))
            if summary and text:
                pairs.append(("xlsum_summary", summary, text[: active().cfg.data.max_text_chars]))
        if unit.limit and len(pairs) >= unit.limit:
            break
    return pairs, dict(st)


def answer_window(context, start, end, width):
    p0 = context.rfind("\n", 0, start) + 1
    p1 = context.find("\n", end)
    p1 = len(context) if p1 < 0 else p1
    while p1 - p0 < width // 3 and p1 < len(context):
        nxt = context.find("\n", p1 + 1)
        p1 = len(context) if nxt < 0 else nxt
    if p1 - p0 <= width:
        return context[p0:p1].strip()
    left = max(p0, start - width // 3)
    if left > p0:
        sentence = max(context.rfind(m, p0, left) for m in (". ", "؟ ", "! "))
        if sentence >= 0 and left - sentence < 300:
            left = sentence + 2
        else:
            space = context.find(" ", left, start)
            left = space + 1 if space >= 0 else left
    right = min(p1, left + width)
    if right < p1:
        space = context.rfind(" ", end, right)
        right = space if space > end else right
    if not (left <= start and end <= right):
        return None
    return context[left:right].strip()


def make_arabicaqa(unit):
    stage = active()
    d = stage.cfg.data
    with open(stage.fetch_file(d.arabicaqa_repo, d.arabicaqa_train, "arabicaqa"), encoding="utf-8") as f:
        train_ids = {str(json.loads(line)["question_id"]) for line in f if line.strip()}
    mrc = json.loads(stage.fetch_file(d.arabicaqa_mrc_repo, d.arabicaqa_mrc_train, "arabicaqa").read_text(encoding="utf-8"))
    pairs, st = [], Counter({"Open-ArabicaQA train questions": len(train_ids)})
    for article in mrc["data"]:
        for par in article["paragraphs"]:
            context = par["context"]
            for qa in par["qas"]:
                if str(qa["id"]) not in train_ids:
                    continue
                st["found in MRC/train.json"] += 1
                answers = qa.get("answers") or []
                text = (answers[0].get("text") or "").strip() if answers else ""
                if qa.get("is_impossible") or not text:
                    st["skipped: no answer"] += 1
                    continue
                start = context.find(text, max(0, answers[0]["answer_start"] - 20))
                start = start if start >= 0 else context.find(text)
                if start < 0:
                    st["skipped: answer not in the article"] += 1
                    continue
                passage = answer_window(context, start, start + len(text), d.arabicaqa_window_chars)
                if passage is None:
                    st["skipped: answer longer than a passage"] += 1
                    continue
                pairs.append(("arabicaqa", qa["question"].strip(), passage))
    if unit.limit:
        pairs = random.Random(d.seed).sample(pairs, min(unit.limit, len(pairs)))
    return pairs, dict(st)


def make_aya(unit):
    stage = active()
    d = stage.cfg.data
    df = pq.read_table(stage.fetch_file(d.aya_repo, d.aya_file, "aya"),
                       columns=["inputs", "targets", "language"]).to_pandas()
    arabic = df[df["language"].str.contains("Arabic", na=False)]
    st = {"rows in the train split": len(df),
          **{f"{k} rows": int(v) for k, v in arabic["language"].value_counts().items()}}
    pairs = [("aya", q.strip(), a.strip()) for q, a in zip(arabic["inputs"], arabic["targets"])
             if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip()]
    if unit.limit:
        pairs = random.Random(d.seed).sample(pairs, min(unit.limit, len(pairs)))
    return pairs, st


QUESTION_LABEL = re.compile(r"^(?:(?:ال)?سؤال|س)\s*\d*\s*[:\-–/)]\s*|^\d{1,3}\s*[-.)]\s*|^[-–•*#>]+\s*")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟…])\s+")


def mined_questions(text):
    d = active().cfg.data
    paras, (lo, hi), out = paragraphs(text), d.qa_mined_words, []
    for i in range(len(paras) - 1):
        if not paras[i].endswith("؟"):
            continue
        question = QUESTION_LABEL.sub("", SENTENCE_SPLIT.split(paras[i])[-1]).strip()
        if not lo <= len(question.split()) <= hi:
            continue
        answer = join_until(paras[i + 1:], d.qa_mined_min_answer_words, stop=lambda p: p.endswith("؟"))
        if not answer or answer.endswith("؟"):
            continue
        out.append((question, answer))
        if len(out) >= d.qa_mined_max_per_doc:
            break
    return out


def make_qa_mined(unit):
    pairs, st = [], Counter()
    for rows in iter_row_groups(unit, ["text"]):
        for r in rows:
            found = mined_questions(r["text"] or "")
            st["documents"] += 1
            st["documents with a pair"] += bool(found)
            pairs += [("qa_mined", q, a) for q, a in found]
        if unit.limit and len(pairs) >= unit.limit:
            break
    return pairs, dict(st)


def pick_span(starts, candidates, length, min_len, limit, rng):
    if not len(candidates):
        return None
    s = int(rng.choice(candidates))
    e = min(s + length, limit)
    if e < limit:
        k = int(np.searchsorted(starts, e, side="right")) - 1
        e = int(starts[k]) if starts[k] > s else e
    return (s, e) if e - s >= min_len else None


def span_pair(text, ids, offsets, rng):
    stage = active()
    n = len(ids)
    q_lo, q_hi = stage.cfg.data.span_query_tokens
    p_lo, p_hi = stage.cfg.data.span_passage_tokens
    q_hi, p_hi = min(q_hi, stage.max_q - 2 - stage.n_q_prefix), min(p_hi, stage.max_p - 2 - stage.n_p_prefix)
    if n < q_lo + p_lo:
        return None
    starts = np.flatnonzero(stage.is_word_start[np.asarray(ids)])
    p_len = min(int(rng.integers(p_lo, p_hi + 1)), n - q_lo)
    room = ((starts >= q_lo) | (starts + p_len + q_lo <= n)) & (starts + p_lo <= n)
    p = pick_span(starts, starts[room], p_len, p_lo, n, rng)
    if p is None:
        return None
    regions = [(a, b) for a, b in ((0, p[0]), (p[1], n)) if b - a >= q_lo]
    if not regions:
        return None
    a, b = regions[int(rng.integers(len(regions)))]
    q = pick_span(starts, starts[(starts >= a) & (starts + q_lo <= b)], int(rng.integers(q_lo, q_hi + 1)), q_lo, b, rng)
    if q is None:
        return None

    def cut(span):
        return text[offsets[span[0]][0]: offsets[span[1] - 1][1]].strip()

    return cut(q), cut(p)


def make_spans(unit):
    stage = active()
    pairs, st, seen = [], Counter(), set()
    for rows in iter_row_groups(unit, ["text"]):
        docs = []
        for r in rows:
            st["documents"] += 1
            text = (r["text"] or "")[:8_000]
            key = hashlib.md5(text.encode("utf-8")).digest()[:8]
            if key in seen:
                st["skipped: duplicate document"] += 1
                continue
            seen.add(key)
            docs.append((text, key))
        enc = stage.tokenizer([t for t, _ in docs], add_special_tokens=False, return_offsets_mapping=True)
        for (text, key), ids, offsets in zip(docs, enc["input_ids"], enc["offset_mapping"]):
            rng = np.random.default_rng([stage.cfg.data.seed, int.from_bytes(key, "little")])
            pair = span_pair(text, ids, offsets, rng)
            if pair is None:
                st["skipped: too short for two spans"] += 1
                continue
            pairs.append(("spans", *pair))
        if unit.limit and len(pairs) >= unit.limit:
            break
    return pairs, dict(st)


MAKERS = {"news": make_news, "wiki": make_wiki, "xlsum": make_xlsum, "arabicaqa": make_arabicaqa, "aya": make_aya,
          "qa_mined": make_qa_mined, "spans": make_spans}


def wiki_heading_check(stage, files):
    check_file = stage.prefer_local(files, "wiki")[0]
    pf = stage.open_parquet(stage.cfg.data.wiki_repo, check_file, "wiki")
    rows = [r for rg in (0, pf.metadata.num_row_groups // 2)
            for r in pf.read_row_group(rg, columns=["title", "text"]).to_pylist()]
    counter, with_heading = Counter(), 0
    for r in rows:
        lines, found = r["text"].split("\n"), False
        for i, raw in enumerate(lines):
            prev = i == 0 or not lines[i - 1].strip()
            nxt = i + 1 == len(lines) or not lines[i + 1].strip()
            if i and raw.strip() and is_heading(raw, prev, nxt):
                found = True
                counter[raw.strip()] += 1
        with_heading += found
    return {"file": check_file, "articles": len(rows), "with_heading": with_heading,
            "most_common": counter.most_common(15)}


def corpus_quality(stage, repo, files, local_name, n_row_groups=3):
    rng, texts = random.Random(0), []
    for f in rng.sample(files, n_row_groups):
        pf = stage.open_parquet(repo, f, local_name, download=False)
        texts += pf.read_row_group(rng.randrange(pf.metadata.num_row_groups), columns=["text"]).column(0).to_pylist()
    single = words = soup = 0
    for t in texts:
        w = t.split()
        words += len(w)
        single += sum(1 for x in w if len(x) == 1 and x != "و" and 0x0600 <= ord(x) <= 0x06FF)
        grams = list(zip(w, w[1:], w[2:]))
        soup += bool(grams) and sum(n for n in Counter(grams).values() if n > 1) / len(grams) > 0.2
    return {"documents": len(texts), "single letters per 1k words": round(single / max(words, 1) * 1000, 1),
            "keyword-soup documents": round(soup / max(len(texts), 1), 3)}


def build_all_sources(stage):
    d = stage.cfg.data
    summaries = {}
    news_files = stage.repo_parquet_files(d.news_repo, "data")
    units = plan_row_group_units(stage, "news", "news", d.news_repo, news_files, "news", d.news_rg_per_unit)
    summaries["news"] = build_source(stage, "news", units)
    report_source(stage, "news", summaries["news"],
                  f"{d.news_repo} ({len(news_files)} parquet files; abuelkhair-corpus/arabic_billion_words has only a "
                  "script and an empty parquet branch)")
    wiki_files = stage.repo_parquet_files(d.wiki_repo, d.wiki_prefix)
    log(f"wiki heading check: {wiki_heading_check(stage, wiki_files)}")
    units = plan_row_group_units(stage, "wiki", "wiki", d.wiki_repo, stage.prefer_local(wiki_files, "wiki"), "wiki",
                                 d.wiki_rg_per_unit)
    summaries["wiki"] = build_source(stage, "wiki", units)
    report_source(stage, "wiki", summaries["wiki"], f"{d.wiki_repo}, config {d.wiki_prefix} (parquet, "
                                                    f"{len(wiki_files)} files)")
    units = plan_row_group_units(stage, "xlsum", "xlsum", d.xlsum_repo, [d.xlsum_file], "xlsum", 10,
                                 revision=d.xlsum_revision)
    summaries["xlsum"] = build_source(stage, "xlsum", units)
    report_source(stage, "xlsum", summaries["xlsum"],
                  f"{d.xlsum_repo}@{d.xlsum_revision}/{d.xlsum_file} (the Hub's parquet conversion)")
    limit = d.smoke_rows // 2 if stage.smoke else None
    units = [Unit("qa", "arabicaqa", "arabicaqa", limit=limit), Unit("qa", "aya", "aya", limit=limit)]
    summaries["qa"] = build_source(stage, "qa", units)
    report_source(stage, "qa", summaries["qa"],
                  f"{d.arabicaqa_repo} {d.arabicaqa_train} (train question ids) + {d.arabicaqa_mrc_repo} "
                  f"{d.arabicaqa_mrc_train} (answer positions in the article); {d.aya_repo} {d.aya_file} (rows whose "
                  "language contains 'Arabic')")
    fineweb_files = stage.repo_parquet_files(d.fineweb2_repo, d.fineweb2_prefix)
    units = plan_row_group_units(stage, "qa_mined", "qa_mined", d.fineweb2_repo,
                                 stage.prefer_local(fineweb_files, "fineweb2"), "fineweb2", d.fineweb2_rg_per_unit)
    summaries["qa_mined"] = build_source(stage, "qa_mined", units)
    report_source(stage, "qa_mined", summaries["qa_mined"],
                  f"{d.fineweb2_repo} {d.fineweb2_prefix} ({len(fineweb_files)} files; replaces the 101B dataset, "
                  "whose text has no punctuation)")
    b101_files = stage.repo_parquet_files(d.b101_repo, d.b101_prefix)
    stage.spans_quality = {
        "101b": corpus_quality(stage, d.b101_repo, stage.prefer_local(b101_files, "101b"), "101b"),
        "fineweb2": corpus_quality(stage, d.fineweb2_repo, stage.prefer_local(fineweb_files, "fineweb2"), "fineweb2"),
    }
    log(f"spans corpus quality: {stage.spans_quality}; spans are built from {d.spans_corpus}")
    if d.spans_corpus == "fineweb2":
        units = plan_row_group_units(stage, "spans", "spans", d.fineweb2_repo,
                                     stage.prefer_local(fineweb_files, "fineweb2"), "fineweb2", d.fineweb2_rg_per_unit)
        loading = f"{d.fineweb2_repo} {d.fineweb2_prefix} (spans_corpus = 'fineweb2')"
    elif stage.smoke:
        units = plan_row_group_units(stage, "spans", "spans", d.b101_repo, stage.prefer_local(b101_files, "101b")[:20],
                                     "101b", 1)
        loading = f"{d.b101_repo} ({len(b101_files)} parquet files)"
    else:
        files = list(b101_files)
        random.Random(f"{d.seed}-spans").shuffle(files)
        units = [Unit("spans", Path(f).stem, "spans", d.b101_repo, f, "101b") for f in files]
        loading = f"{d.b101_repo} ({len(b101_files)} parquet files)"
    summaries["spans"] = build_source(stage, "spans", units)
    report_source(stage, "spans", summaries["spans"], loading)
    write_json(stage.loading, stage.raw / "loading.json")
    return summaries


def raw_summary(stage):
    return {name: json.loads((stage.raw / name / "summary.json").read_text(encoding="utf-8"))
            for name in stage.source_names}
