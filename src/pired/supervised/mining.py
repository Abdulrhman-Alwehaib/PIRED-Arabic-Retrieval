import dataclasses
import hashlib
import math
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..common.device import device_name, free_gpu
from ..common.io import read_json, write_json, write_parquet, write_text
from ..common.runlog import log
from ..common.text import fmt_hours, md_table, short
from ..contrastive.steps import clean_text
from ..encoder.embedding import pad_chunks
from .config import MiningSettings
from .queries import wiki_passage

DOMAINS = ["wiki", "news", "mmarco"]
NEG_ORIGINS = ("judged", "top15", "top16-50", "random")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def cap_synthetic(q, cap, per_passage, seed):
    rng = np.random.default_rng(seed)
    q = q.assign(_k=rng.random(len(q)))
    search = (q["qtype"] == "search_query").to_numpy()
    first_full = np.zeros(len(q), dtype=bool)
    full_rank = q[~search].groupby("group")["_k"].rank(method="first").to_numpy()
    first_full[np.flatnonzero(~search)[full_rank == 1]] = True
    q = q.assign(_extra=~(search | first_full)).sort_values(["group", "_extra", "_k"])
    q["_rank"] = q.groupby("group").cumcount()
    q = q[q["_rank"] < per_passage]
    if len(q) > cap:
        size = q.groupby("group").size()
        order = size.to_frame("n").assign(_r=rng.random(len(size))).sort_values(["n", "_r"])
        drop = order.index[: np.searchsorted(order["n"].cumsum().to_numpy(), len(q) - cap) + 1]
        q = q[~q["group"].isin(drop)]
    return q.drop(columns=["_k", "_extra", "_rank"]).sort_index()


def cap_counts(stage, available):
    f = stage.cfg.filters
    n = {s: available[s] if (stage.smoke or stage.sources[s].target is None) else min(available[s],
                                                                                        stage.sources[s].target)
         for s in available}
    caps = {s: (f.mmarco_max_share if s == "mmarco" else f.max_share) for s in n}
    for _ in range(100):
        total = sum(n.values())
        over = [s for s in n if n[s] > caps[s] * total + 1e-9]
        if not over:
            break
        for s in over:
            n[s] = int(caps[s] / (1 - caps[s]) * (total - n[s]))
    return n


class Pool:
    def __init__(self, stage, domain):
        self.stage, self.domain = stage, domain
        self.shards = sorted((stage.p2 / "pools" / domain).glob("shard-*.parquet"))
        self.pids = pa.concat_tables([pq.read_table(s, columns=["pid"]) for s in self.shards])["pid"].to_numpy(
            zero_copy_only=False)
        self.offsets = np.cumsum([0] + [pq.ParquetFile(s).metadata.num_rows for s in self.shards])
        self._texts = None

    def __len__(self):
        return len(self.pids)

    def texts(self, rows):
        if self._texts is None:
            self._texts = pa.concat_tables([pq.read_table(s, columns=["text"]) for s in self.shards])["text"].cast(
                pa.large_string()).combine_chunks()
        return self._texts.take(pa.array(np.asarray(rows, dtype=np.int64))).to_pylist()

    def emb_file(self, k):
        return self.stage.p2 / "embeddings" / self.domain / f"shard-{k:05d}.npy"


class Miner:
    def __init__(self, stage):
        self.stage, self.cfg = stage, stage.cfg
        self.p2 = stage.p2
        self.stats = {}
        self.in1 = stage.pull_folder(stage.repo_id("queries"), "dataset", stage.final1)
        self.q1 = pd.read_parquet(self.in1 / "queries.parquet")
        self.pass1 = pd.read_parquet(self.in1 / "passages.parquet").set_index("pid")
        in_fp = hashlib.sha256((self.in1 / "queries.parquet").read_bytes()
                               + (self.in1 / "passages.parquet").read_bytes()).hexdigest()[:16]
        fp_file = self.p2 / "input_fingerprint.txt"
        if fp_file.exists() and fp_file.read_text().strip() != in_fp:
            for d in ("pools", "embeddings", "candidates", "rerank"):
                shutil.rmtree(self.p2 / d, ignore_errors=True)
            log("Part 1's output changed since the last mining run: pools, embeddings, candidates and scores are rebuilt")
        self.p2.mkdir(parents=True, exist_ok=True)
        fp_file.write_text(in_fp)
        log(f"Part 1 output: {len(self.q1):,} queries ({dict(self.q1['split'].value_counts())}), "
            f"{len(self.pass1):,} passages; queries per domain: {dict(self.q1['domain'].value_counts())}")
        self.pools = {}

    def apply_caps(self):
        stage, caps = self.stage, self.cfg.mining.rerank_caps
        q1, train = self.q1, self.q1["split"] == "train"
        before = q1[train]["source"].value_counts().to_dict()
        parts = [q1[~(train & q1["source"].isin(caps))]]
        for s, cap in caps.items():
            q = q1[train & (q1["source"] == s)]
            if s == "synthetic":
                q = cap_synthetic(q, cap, self.cfg.gen.max_questions_per_passage, self.cfg.data.seed)
            elif len(q) > cap:
                q = q.sample(n=cap, random_state=self.cfg.data.seed)
            parts.append(q)
        self.q1 = pd.concat(parts).sort_index().reset_index(drop=True)
        after = self.q1[self.q1["split"] == "train"]["source"].value_counts().to_dict()
        self.stats["caps"] = {"settings": caps, "train_before": before, "train_after": after}
        lines = [f"{'source':<11}{'train before':>14}{'train after':>13}{'validation':>12}"]
        for s in stage.source_names:
            n_val = int(((self.q1["source"] == s) & (self.q1["split"] == "validation")).sum())
            lines.append(f"{s:<11}{before.get(s, 0):>14,}{after.get(s, 0):>13,}{n_val:>12,}")
        lines.append(f"{'total':<11}{sum(before.values()):>14,}{sum(after.values()):>13,}")
        log("\n".join(lines))
        q_fp = hashlib.sha256("\n".join(self.q1["qid"]).encode("utf-8")).hexdigest()[:16]
        q_fp_file = self.p2 / "queries_fingerprint.txt"
        if q_fp_file.exists() and q_fp_file.read_text().strip() != q_fp:
            for d in ("candidates", "rerank"):
                shutil.rmtree(self.p2 / d, ignore_errors=True)
            log("the mined queries changed since the last run: candidates and reranker scores are rebuilt")
        q_fp_file.write_text(q_fp)

    def write_pool(self, domain, pids, texts):
        d = self.p2 / "pools" / domain
        shutil.rmtree(d, ignore_errors=True)
        df = pd.DataFrame({"pid": pids, "text": texts}).drop_duplicates("pid").reset_index(drop=True)
        n = self.cfg.mining.pool_shard_rows
        for k in range(math.ceil(len(df) / n)):
            write_parquet(df.iloc[k * n: (k + 1) * n], d / f"shard-{k:05d}.parquet")
        info = {"passages": len(df), "shards": math.ceil(len(df) / n)}
        write_json(info, d / "info.json")
        return info

    def build_pools(self):
        stage, m = self.stage, self.cfg.mining
        info = {}
        part1 = {dom: self.pass1[self.pass1["domain"] == dom]["text"] for dom in DOMAINS}
        for domain in DOMAINS:
            f = self.p2 / "pools" / domain / "info.json"
            if f.exists():
                info[domain] = read_json(f)
                log(f"{domain}: pool built earlier ({info[domain]['passages']:,} passages)")
                continue
            t0 = time.time()
            if domain == "wiki":
                corpus = stage.miracl_corpus_frame()
                raw = [wiki_passage(t, x) for t, x in zip(corpus["title"], corpus["text"])]
                source = f"MIRACL-ar corpus ({len(raw):,}{', SMOKE subset' if stage.smoke else ''})"
            elif domain == "news":
                leads, source = stage.news_leads(m.news_pool_size)
                raw = leads["text"].tolist()[: m.news_pool_size]
            else:
                raw = [t for _, t in stage.read_mmarco_collection(
                    max_bytes=self.cfg.data.mmarco_smoke_bytes if stage.smoke else None)]
                source = f"mMARCO Arabic collection ({len(raw):,}{', SMOKE: start of the file' if stage.smoke else ''})"
            pids = [f"{domain}:{stage.text_key(t)}" for t in raw]
            texts = [clean_text(t)[0] for t in raw]
            mine = part1[domain]
            pid_set = set(pids)
            extra = [p for p in mine.index if p not in pid_set]
            known = dict(zip(pids, texts))
            known.update(mine.to_dict())
            all_pids = list(dict.fromkeys(pids + extra))
            info[domain] = {**self.write_pool(domain, all_pids, [known[p] for p in all_pids]), "from": source,
                            "part1_passages_added": len(extra), "seconds": time.time() - t0}
            write_json(info[domain], f)
            log(f"{domain}: {info[domain]['passages']:,} passages from {source}, + {len(extra):,} Part-1 passages that "
                f"were not in it ({info[domain]['seconds']:.0f}s)")
        self.stats["pools"] = info
        self.pools = {d: Pool(stage, d) for d in DOMAINS}
        return info

    def encode_shard(self, pool, k):
        out = pool.emb_file(k)
        if out.exists():
            return 0, 0.0
        texts = pd.read_parquet(pool.shards[k], columns=["text"])["text"].tolist()
        t0 = time.time()
        embedder = self.stage.embedder
        ids = embedder.tokenize(texts, "passage")
        emb = embedder.embed_ids(ids, self.cfg.mining.encode_batch_tokens, dtype=torch.float16).cpu().numpy()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out.with_suffix(".tmp.npy"), emb)
        os.replace(out.with_suffix(".tmp.npy"), out)
        return sum(map(len, ids)), time.time() - t0

    def encode_pools(self):
        stage, m = self.stage, self.cfg.mining
        embedder = stage.embedder
        stage.encoder.eval()
        sample = self.pools["wiki"].texts(range(min(2_000, len(self.pools["wiki"]))))
        t0 = time.time()
        ids = embedder.tokenize(sample, "passage")
        embedder.embed_ids(ids, m.encode_batch_tokens, dtype=torch.float16)
        if stage.device.type == "cuda":
            torch.cuda.synchronize()
        rate = sum(map(len, ids)) / (time.time() - t0)
        tok_per_passage = sum(map(len, ids)) / len(ids)
        full_pool = ((2_064_310 + MiningSettings().news_pool_size + 8_841_823) if stage.smoke
                     else sum(len(p) for p in self.pools.values()))
        stage.timings["encode"] = {"tokens_per_s": rate, "tokens_per_passage": tok_per_passage}
        log(f"sample: {len(ids):,} passages, {tok_per_passage:.0f} tokens each, {rate:,.0f} tokens/s on "
            f"{device_name(stage.device)}; estimate for the full pools ({full_pool:,} passages): "
            f"~{fmt_hours(full_pool * tok_per_passage / rate)} on this GPU")
        for d, pool in self.pools.items():
            tokens, seconds = 0, 0.0
            for k in range(len(pool.shards)):
                a, b = self.encode_shard(pool, k)
                tokens, seconds = tokens + a, seconds + b
                if a and not stage.smoke:
                    stage.announce(f"encoded {d} shard {k + 1}/{len(pool.shards)} ({a / max(b, 1e-9):,.0f} tokens/s)")
            log(f"{d}: {len(pool):,} passages encoded"
                + (f" in {seconds:.0f}s ({tokens / max(seconds, 1e-9):,.0f} tokens/s)" if seconds else " (earlier)"))

    @torch.no_grad()
    def search_domain(self, domain, q):
        cand_dir = self.p2 / "candidates"
        out = cand_dir / f"{domain}.parquet"
        if out.exists():
            return pd.read_parquet(out)
        stage, pool, m = self.stage, self.pools[domain], self.cfg.mining
        device = stage.device
        k_extra = m.top_k + 10
        t0 = time.time()
        qv = stage.embedder.embed_ids(stage.embedder.tokenize(q["query"].tolist(), "query"), m.encode_batch_tokens,
                                      dtype=torch.float16)
        best_s = torch.full((len(q), k_extra), -1e4, dtype=torch.float32, device=device)
        best_i = torch.full((len(q), k_extra), -1, dtype=torch.int64, device=device)
        for k in range(len(pool.shards)):
            pv = torch.from_numpy(np.load(pool.emb_file(k))).to(device)
            base = int(pool.offsets[k])
            kk = min(k_extra, len(pv))
            for a in range(0, len(q), m.search_query_chunk):
                b = min(a + m.search_query_chunk, len(q))
                s, i = (qv[a:b] @ pv.T).float().topk(kk, dim=1)
                s = torch.cat([best_s[a:b], s], 1)
                i = torch.cat([best_i[a:b], i + base], 1)
                top = s.topk(k_extra, dim=1)
                best_s[a:b], best_i[a:b] = top.values, i.gather(1, top.indices)
            del pv
        rows, scores = best_i.cpu().numpy(), best_s.cpu().numpy()
        cand, cand_s, n_pos_found = [], [], 0
        for r, sc, pos in zip(rows, scores, q["pos_pids"]):
            pos = set(pos)
            keep = [(int(x), float(y)) for x, y in zip(r, sc) if x >= 0 and pool.pids[x] not in pos]
            n_pos_found += sum(1 for x in r if x >= 0 and pool.pids[x] in pos)
            cand.append([x for x, _ in keep[: m.top_k]])
            cand_s.append([round(y, 4) for _, y in keep[: m.top_k]])
        df = pd.DataFrame({"qid": q["qid"].to_numpy(), "cand": cand, "cand_dense": cand_s})
        write_parquet(df, out)
        write_json({"queries": len(q), "seconds": time.time() - t0, "positives_in_top60": n_pos_found},
                   cand_dir / f"{domain}.json")
        return df

    def search(self):
        cands = {}
        for d in DOMAINS:
            q = self.q1[self.q1["domain"] == d]
            if not len(q):
                continue
            cands[d] = self.search_domain(d, q)
            info = read_json(self.p2 / "candidates" / f"{d}.json")
            hit = info["positives_in_top60"] / max(1, q["pos_pids"].map(len).sum())
            log(f"{d}: {len(q):,} queries searched in {len(self.pools[d]):,} passages ({info['seconds']:.0f}s); "
                f"{hit:.1%} of the positives were in the stage-3 top 60 (removed from the candidates)")
        self.cand = pd.concat(cands.values(), ignore_index=True).set_index("qid")

    def load_reranker(self):
        m = self.cfg.mining
        self.rerank_tok = AutoTokenizer.from_pretrained(m.reranker)
        self.reranker = AutoModelForSequenceClassification.from_pretrained(
            m.reranker, dtype=torch.bfloat16).to(self.stage.device).eval()

    @torch.no_grad()
    def rerank_pairs(self, queries, passages):
        m, device = self.cfg.mining, self.stage.device
        out, n_tokens, block = np.empty(len(queries), dtype=np.float32), 0, 32_768
        with ThreadPoolExecutor(1) as pool:
            def submit(a):
                return pool.submit(self.rerank_tok, queries[a: a + block], passages[a: a + block],
                                   truncation="only_second", max_length=m.reranker_max_len)

            future = submit(0)
            for a in range(0, len(queries), block):
                ids = future.result()["input_ids"]
                if a + block < len(queries):
                    future = submit(a + block)
                n_tokens += sum(map(len, ids))
                for rows, x, mask in pad_chunks(ids, m.rerank_batch_tokens, self.rerank_tok.pad_token_id):
                    x = x.to(device)
                    mask = torch.ones_like(x) if mask is None else mask.to(device)
                    logits = self.reranker(input_ids=x, attention_mask=mask).logits[:, 0].float()
                    out[(rows + a).numpy()] = logits.cpu().numpy()
        return out, n_tokens

    def rerank_queries(self, q):
        m = self.cfg.mining
        qs, ps, owner = [], [], []
        for i, r in enumerate(q.itertuples()):
            pos = [self.pass1.at[p, "text"] for p in r.pos_pids[: m.max_positives_scored]]
            cand = self.pools[r.domain].texts(self.cand.at[r.qid, "cand"][: m.rerank_top])
            for kind, texts in (("p", pos), ("c", cand)):
                qs += [r.query] * len(texts)
                ps += texts
                owner += [(i, kind)] * len(texts)
        t0 = time.time()
        scores, n_tokens = self.rerank_pairs(qs, ps)
        seconds = time.time() - t0
        pos_s, cand_s = [[] for _ in range(len(q))], [[] for _ in range(len(q))]
        for (i, kind), s in zip(owner, scores):
            (pos_s if kind == "p" else cand_s)[i].append(round(float(s), 4))
        return pd.DataFrame({"qid": q["qid"].to_numpy(), "pos_scores": pos_s, "cand_scores": cand_s}), n_tokens, seconds

    def rerank(self):
        stage, m = self.stage, self.cfg.mining
        self.load_reranker()
        sample = self.q1.sample(min(200, len(self.q1)), random_state=0)
        _, tokens, seconds = self.rerank_queries(sample)
        pairs = sum(min(len(p), m.max_positives_scored) for p in sample["pos_pids"]) + m.rerank_top * len(sample)
        stage.timings["rerank"] = {"pairs_per_s": pairs / seconds, "tokens_per_pair": tokens / pairs,
                                   "tokens_per_s": tokens / seconds}
        full_q = 1_700_000 if stage.smoke else len(self.q1)
        log(f"sample: {len(sample)} queries = {pairs:,} pairs, {tokens / pairs:.0f} XLM-R tokens each, "
            f"{pairs / seconds:,.0f} pairs/s on {device_name(stage.device)}; estimate for "
            f"{'a full run (~1.7M queries)' if stage.smoke else f'all {len(self.q1):,} queries'}: "
            f"~{fmt_hours(full_q * pairs / len(sample) / (pairs / seconds))}")
        rerank_dir = self.p2 / "rerank"
        n = m.rerank_shard_queries
        n_shards = math.ceil(len(self.q1) / n)
        done0 = sum((rerank_dir / f"shard-{k:05d}.parquet").exists() for k in range(n_shards))
        stage.announce(f"reranking: {len(self.q1):,} queries in {n_shards} shards, {done0} done earlier")
        secs, new = 0.0, 0
        for k in range(n_shards):
            out = rerank_dir / f"shard-{k:05d}.parquet"
            if out.exists():
                continue
            df, _, s = self.rerank_queries(self.q1.iloc[k * n: (k + 1) * n])
            write_parquet(df, out)
            secs, new = secs + s, new + 1
            stage.announce(f"rerank shard {k + 1}/{n_shards}: {s:.0f}s; ETA ~{fmt_hours((n_shards - k - 1) * secs / new)}")
        self.scores = pd.concat([pd.read_parquet(p) for p in sorted(rerank_dir.glob("shard-*.parquet"))],
                                ignore_index=True).set_index("qid")
        self.stats["rerank"] = {"queries": len(self.scores), **stage.timings["rerank"]}
        self.reranker = None
        free_gpu()

    def false_negatives(self):
        ratio = self.cfg.mining.false_negative_ratio
        mq = self.q1.set_index("qid").join(self.scores, how="inner").join(self.cand, how="inner")
        mq["best_pos"] = [float(sigmoid(max(s))) if len(s) else float("nan") for s in mq["pos_scores"]]
        mq["cand_prob"] = pd.Series([sigmoid(s) for s in mq["cand_scores"]], index=mq.index, dtype=object)
        mq["false_neg"] = pd.Series([p >= ratio * b for p, b in zip(mq["cand_prob"], mq["best_pos"])], index=mq.index,
                                    dtype=object)
        fn = mq.groupby("source")["false_neg"].apply(lambda s: int(sum(x.sum() for x in s)))
        scored = mq.groupby("source")["cand_prob"].apply(lambda s: int(sum(len(x) for x in s)))
        self.stats["false_negatives"] = {s: {"candidates_scored": int(scored[s]), "false_negatives": int(fn[s])}
                                         for s in fn.index}
        lines = [f"{'source':<11}{'reranked candidates':>21}{'false negatives':>17}{'share':>8}"]
        lines += [f"{s:<11}{scored[s]:>21,}{fn[s]:>17,}{fn[s] / max(scored[s], 1):>8.1%}" for s in fn.index]
        lines.append(f"examples (threshold: >= {ratio:.0%} of the best positive's probability):")
        flagged = mq[mq["false_neg"].map(lambda x: x.any())].sample(frac=1, random_state=1)
        for _, r in flagged.head(5).iterrows():
            j = int(np.flatnonzero(r["false_neg"])[0])
            lines.append(f"  Q [{r['source']}]: {short(r['query'], 110)}")
            lines.append(f"     positive  (p={r['best_pos']:.3f}): {short(self.pass1.at[r['pos_pids'][0], 'text'], 140)}")
            lines.append(f"     dropped   (p={r['cand_prob'][j]:.3f}): "
                         f"{short(self.pools[r['domain']].texts([r['cand'][j]])[0], 140)}")
        log("\n".join(lines))
        self.mq = mq

    def consistency(self):
        stage, m, mq = self.stage, self.cfg.mining, self.mq
        mq["pos_rank"] = [1 + int((sigmoid(c) > b).sum()) for c, b in zip(mq["cand_scores"], mq["best_pos"])]
        top = mq["source"].map(lambda s: m.consistency_top_by_source.get(s, m.consistency_top)).to_numpy()
        mq["consistent"] = mq["pos_rank"].to_numpy() <= top
        rows = []
        for s, g in mq.groupby("source", sort=False):
            fails = int((~g["consistent"]).sum())
            rows.append([s, len(g), fails, "kept (human-labeled)" if stage.sources[s].human else "removed",
                         dict(Counter(np.minimum(g["pos_rank"], 5)).most_common())])
        self.stats["consistency"] = {r[0]: {"queries": r[1], "rank_above_cutoff": r[2], "action": r[3]} for r in rows}
        lines = [f"{'source':<11}{'queries':>9}{'failed':>10}   action   rank histogram (5 = 5 or worse)"]
        lines += [f"{r[0]:<11}{r[1]:>9,}{r[2]:>10,}   {r[3]:<21}{r[4]}" for r in rows]
        drop = ~mq["consistent"] & ~mq["is_human"]
        lines.append("examples removed:")
        for _, r in mq[drop].sample(min(5, int(drop.sum())), random_state=2).iterrows():
            j = int(np.argmax(r["cand_prob"]))
            lines.append(f"  Q [{r['source']}] rank {r['pos_rank']}: {short(r['query'], 110)}")
            lines.append(f"     positive (p={r['best_pos']:.3f}): {short(self.pass1.at[r['pos_pids'][0], 'text'], 130)}")
            lines.append(f"     best candidate (p={r['cand_prob'][j]:.3f}): "
                         f"{short(self.pools[r['domain']].texts([r['cand'][j]])[0], 130)}")
        log("\n".join(lines))
        self.mq = mq[~drop]

    def final_mix(self):
        stage, cfg, mq = self.stage, self.cfg, self.mq
        syn = mq["source"] == "synthetic"
        order = mq[syn].sample(frac=1, random_state=cfg.data.seed)
        keep_syn = order.groupby("group").cumcount() < cfg.gen.max_questions_per_passage
        cut = int((~keep_syn).sum())
        mq = pd.concat([mq[~syn], order[keep_syn]])
        log(f"synthetic: {cut:,} questions removed to keep at most {cfg.gen.max_questions_per_passage} per passage")
        train = mq[mq["split"] == "train"]
        avail = {s: int((train["source"] == s).sum()) for s in stage.source_names}
        counts = cap_counts(stage, avail)
        chosen = []
        for i, s in enumerate(stage.source_names):
            g = train[train["source"] == s]
            chosen.append(g.sample(n=counts[s], random_state=cfg.data.seed + i) if counts[s] < len(g) else g)
        self.train_q = pd.concat(chosen)
        self.val_q = mq[mq["split"] == "validation"]
        total = sum(counts.values())
        self.mix = {s: {"available": avail[s], "target": stage.sources[s].target, "final": counts[s],
                        "share": round(counts[s] / max(total, 1), 4),
                        "rows": counts[s] * (cfg.filters.repeats_human if stage.sources[s].human else 1),
                        "validation": int((self.val_q["source"] == s).sum()), "license": stage.sources[s].license}
                    for s in stage.source_names}
        lines = [f"{'source':<11}{'available':>11}{'target':>10}{'final':>10}{'share':>8}{'train rows':>12}{'validation':>12}"]
        for s, v in self.mix.items():
            target = f"{v['target']:,}" if v["target"] and not stage.smoke else "all"
            lines.append(f"{s:<11}{v['available']:>11,}{target:>10}{v['final']:>10,}{v['share']:>8.1%}{v['rows']:>12,}"
                         f"{v['validation']:>12,}")
        log("\n".join(lines))
        self.stats["final_mix"] = self.mix
        self.stats["synthetic_cut_to_2_per_passage"] = cut
        self.mq = mq

    def pick_negatives(self, r, rng):
        stage, m, pool = self.stage, self.cfg.mining, self.pools[r["domain"]]
        pos_keys = {stage.text_key(self.pass1.at[p, "text"]) for p in r["pos_pids"]}
        texts, probs, origin = [], [], []

        def take(cands, kind):
            for text, prob in cands:
                if len(texts) == m.n_negatives:
                    return
                if stage.text_key(text) in pos_keys or text in texts:
                    continue
                texts.append(text)
                probs.append(prob)
                origin.append(kind)

        judged = [self.pass1.at[p, "text"] for p in r["neg_pids"] if p in self.pass1.index]
        take([(judged[i], float("nan")) for i in rng.permutation(len(judged))], "judged")
        top = [j for j in range(min(m.rerank_top, len(r["cand"]))) if not r["false_neg"][j]]
        top = [top[i] for i in rng.permutation(len(top))]
        take(zip(pool.texts([r["cand"][j] for j in top]), [float(r["cand_prob"][j]) for j in top]), "top15")
        rest = list(range(m.rerank_top, len(r["cand"])))
        rest = [rest[i] for i in rng.permutation(len(rest))]
        take(zip(pool.texts([r["cand"][j] for j in rest]), [float("nan")] * len(rest)), "top16-50")
        while len(texts) < m.n_negatives:
            take(zip(pool.texts(rng.integers(0, len(pool), 4 * m.n_negatives)), [float("nan")] * 4 * m.n_negatives),
                 "random")
        return texts, probs, origin

    def build_training_rows(self, q):
        cfg, rows = self.cfg, []
        for qid, r in q.iterrows():
            reps = cfg.filters.repeats_human if r["is_human"] else 1
            pos_texts = [self.pass1.at[p, "text"] for p in r["pos_pids"]]
            pos_probs = sigmoid(r["pos_scores"]) if len(r["pos_scores"]) else np.full(len(pos_texts), np.nan)
            for rep in range(reps):
                rng = np.random.default_rng([cfg.data.seed, int(hashlib.md5(qid.encode()).hexdigest()[:8], 16), rep])
                texts, probs, origin = self.pick_negatives(r, rng)
                first = rep % len(pos_texts)
                order = [first] + [i for i in range(len(pos_texts)) if i != first]
                rows.append({"source": r["source"], "query": r["query"], "positives": [pos_texts[i] for i in order],
                             "negatives": texts,
                             "pos_score": float(pos_probs[first]) if first < len(pos_probs) else float("nan"),
                             "neg_scores": probs, "is_human_labeled": bool(r["is_human"]), "qid": qid, "repeat": rep,
                             "domain": r["domain"], "neg_origin": origin})
        return pd.DataFrame(rows)

    def write_training_rows(self):
        stage, final2 = self.stage, self.stage.final2
        shutil.rmtree(final2, ignore_errors=True)
        origin, n_rows, t0 = Counter(), {}, time.time()
        for s in stage.source_names:
            g = self.train_q[self.train_q["source"] == s].sample(frac=1, random_state=self.cfg.data.seed)
            count = 0
            for k, a in enumerate(range(0, len(g), 100_000)):
                df = self.build_training_rows(g.iloc[a: a + 100_000])
                for o in df["neg_origin"]:
                    origin.update(o)
                df = df.sample(frac=1, random_state=k)
                write_parquet(df, final2 / "train" / s / f"part-{k:05d}.parquet")
                count += len(df)
            n_rows[s] = count
        log(f"training rows written in {time.time() - t0:.0f}s: {n_rows}; where the negatives come from: "
            + ", ".join(f"{k} {v:,} ({v / max(sum(origin.values()), 1):.1%})" for k, v in origin.items()))
        self.stats["negatives_origin"] = dict(origin)
        self.stats["train_rows"] = n_rows
        self.n_rows = n_rows

    def no_positive_among_negatives(self):
        stage, n_neg = self.stage, self.cfg.mining.n_negatives
        rows_checked = negs = viol = wrong_n = 0
        for f in sorted((stage.final2 / "train").rglob("*.parquet")):
            df = pd.read_parquet(f)
            rows_checked += len(df)
            negs += int(df["negatives"].map(len).sum())
            wrong_n += int((df["negatives"].map(len) != n_neg).sum())
            viol += sum(len({stage.text_key(t) for t in neg} & {stage.text_key(t) for t in pos})
                        for pos, neg in zip(df["positives"], df["negatives"]))
        if viol or wrong_n:
            raise AssertionError(f"{viol} negatives equal a positive, {wrong_n} rows without {n_neg} negatives")
        return (f"PASS {rows_checked:,} training rows, {negs:,} negatives: none has the text of one of its own "
                f"positives; every row has {n_neg} negatives")

    def build_validation(self):
        stage, cfg = self.stage, self.cfg
        corpus, qrows = {}, []

        def add(pid, domain, text):
            if pid not in corpus:
                corpus[pid] = (len(corpus), domain, text)
            return corpus[pid][0]

        for qid, r in self.val_q.iterrows():
            pos = [add(p, r["domain"], self.pass1.at[p, "text"]) for p in r["pos_pids"]]
            keep = [c for j, c in enumerate(r["cand"]) if not (j < len(r["false_neg"]) and r["false_neg"][j])]
            for pid, text in zip(self.pools[r["domain"]].pids[keep], self.pools[r["domain"]].texts(keep)):
                add(pid, r["domain"], text)
            qrows.append({"qid": qid, "source": r["source"], "domain": r["domain"], "query": r["query"],
                          "positives": pos})
        shares = self.val_q["domain"].value_counts(normalize=True)
        rng = np.random.default_rng(cfg.data.seed)
        for dom, share in shares.items():
            pool = self.pools[dom]
            rows = rng.choice(len(pool), min(len(pool), int(round(cfg.mining.val_random_passages * share))),
                              replace=False)
            for pid, text in zip(pool.pids[rows], pool.texts(rows)):
                add(pid, dom, text)
        cdf = pd.DataFrame([(i, pid, d, t) for pid, (i, d, t) in corpus.items()], columns=["cid", "pid", "domain", "text"])
        write_parquet(pd.DataFrame(qrows), stage.final2 / "validation" / "queries.parquet")
        write_parquet(cdf.sort_values("cid"), stage.final2 / "validation" / "corpus.parquet")
        return {"queries": len(qrows), "corpus": len(cdf)}

    def save(self, test4):
        stage, cfg, final2 = self.stage, self.cfg, self.stage.final2
        val_info = self.build_validation()
        self.stats["validation"] = val_info
        manifest = {"part": 2, "smoke": stage.smoke, "train_rows": self.n_rows, "validation": val_info,
                    "columns": ["source", "query", "positives", "negatives", "pos_score", "neg_scores",
                                "is_human_labeled", "qid", "repeat", "domain", "neg_origin"],
                    "n_negatives": cfg.mining.n_negatives, "prefixes": {"query": stage.q_prefix, "passage": stage.p_prefix},
                    "tokenizer_fingerprint": stage.tokenizer_fingerprint, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        write_json(manifest, final2 / "manifest.json")
        write_json({"mining": self.stats, "settings": dataclasses.asdict(cfg.mining), "test4": test4,
                    "timings": {k: stage.timings[k] for k in ("encode", "rerank") if k in stage.timings}},
                   final2 / "stats.json")
        if (self.in1 / "stats.json").exists():
            shutil.copyfile(self.in1 / "stats.json", final2 / "part1_stats.json")
        write_text(f"# Stage 4 training data: queries with hard negatives{' (SMOKE)' if stage.smoke else ''}\n\n"
                   f"Built by `pired.supervised` (Part 2) on {time.strftime('%Y-%m-%d')} from the queries of "
                   f"`{stage.repo_id('queries')}`.\n\n"
                   + md_table(["source", "final queries", "share", "training rows", "validation"],
                              [[s, v["final"], f"{v['share']:.1%}", v["rows"], v["validation"]]
                               for s, v in self.mix.items()])
                   + "\n\nColumns: " + ", ".join(f"`{c}`" for c in manifest["columns"])
                   + ". `neg_scores`: reranker probability (NaN = not reranked). Validation: "
                     "`validation/queries.parquet` (positives = corpus ids) and `validation/corpus.parquet`.\n",
                   final2 / "README.md")
        log(f"validation: {val_info['queries']:,} queries, {val_info['corpus']:,} passages in their corpus; saved to "
            f"{final2}")
        stage.push_folder(final2, stage.repo_id("pairs"), "dataset")
        return manifest

    def run(self):
        self.apply_caps()
        self.build_pools()
        if self.stage.encoder is None:
            self.stage.load_encoder()
        self.encode_pools()
        self.search()
        self.rerank()
        self.false_negatives()
        self.consistency()
        self.final_mix()
        self.write_training_rows()
        test4 = self.no_positive_among_negatives()
        log(test4)
        return self.save(test4)


def example_rows(stage, n=20):
    examples = []
    for k, f in enumerate(sorted((stage.final2 / "train").rglob("*.parquet"))):
        df = pd.read_parquet(f)
        examples.append(df.sample(min(len(df), n), random_state=k))
    ex = pd.concat(examples, ignore_index=True)
    lines = []
    for i, r in enumerate(ex.sample(min(n, len(ex)), random_state=5).itertuples(), 1):
        def fmt(p):
            return "-" if p != p else f"{p:.3f}"

        lines.append(f"{i:>2}. [{r.source}{', repeat ' + str(r.repeat) if r.is_human_labeled else ''}] Q: "
                     f"{short(r.query, 120)}")
        lines.append(f"    + ({fmt(r.pos_score)}) {short(r.positives[0], 170)}")
        lines += [f"    - ({fmt(p)}, {o}) {short(t, 170)}"
                  for t, p, o in list(zip(r.negatives, r.neg_scores, r.neg_origin))[:3]]
    return "\n".join(lines)
