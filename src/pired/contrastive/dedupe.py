import hashlib
import time

import numpy as np
import pandas as pd

from ..common.parallel import parallel_map
from ..common.text import short
from .steps import (JOB, WORD_RE, clean_shards, fetch_texts, finish_source, load_alive, save_step, stable_hash,
                    step_done, take_sample)

MIX = np.uint64(0x9E3779B97F4A7C15)


class MinHasher:
    def __init__(self, seed, perms, ngram):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 2**62, size=perms, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
        self.b = rng.integers(0, 2**62, size=perms, dtype=np.uint64)
        self.ngram = ngram

    @staticmethod
    def word_hash(word):
        return int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "little")

    def shingle_hashes(self, text, cache):
        words = WORD_RE.findall(text)
        if not words:
            return np.zeros(1, dtype=np.uint64)
        wh = np.fromiter((cache[w] if w in cache else cache.setdefault(w, self.word_hash(w)) for w in words),
                         dtype=np.uint64, count=len(words))
        n = min(self.ngram, len(wh))
        h = wh[: len(wh) - n + 1].copy()
        for k in range(1, n):
            h = h * MIX + wh[k: len(wh) - n + 1 + k]
        return np.unique(h)

    def signatures(self, texts, budget=32_000_000):
        cache, k_perms = {}, len(self.a)
        sh = [self.shingle_hashes(t, cache) for t in texts]
        lens = np.array([len(s) for s in sh], dtype=np.int64)
        flat = np.concatenate(sh) if sh else np.zeros(0, np.uint64)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        sig = np.empty((len(texts), k_perms), dtype=np.uint32)
        i = 0
        while i < len(texts):
            j, total = i, 0
            while j < len(texts) and (j == i or (total + lens[j]) * k_perms <= budget):
                total += lens[j]
                j += 1
            seg = flat[starts[i]: starts[i] + total]
            vals = ((seg[:, None] * self.a[None, :] + self.b[None, :]) >> np.uint64(32)).astype(np.uint32)
            sig[i:j] = np.minimum.reduceat(vals, starts[i:j] - starts[i], axis=0)
            i = j
        return sig


def make_minhasher(stage):
    f = stage.cfg.filters
    return MinHasher(stage.cfg.data.seed, f.minhash_perms, f.minhash_ngram)


def near_duplicate_keepers(sig, bands, threshold):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n, k_perms = sig.shape
    r, codes = k_perms // bands, []
    for b in range(bands):
        key = sig[:, b * r].astype(np.uint64)
        for c in range(1, r):
            key = key * MIX + sig[:, b * r + c].astype(np.uint64)
        order = np.argsort(key, kind="stable")
        sk = key[order]
        new = np.concatenate([[True], sk[1:] != sk[:-1]])
        group = np.cumsum(new) - 1
        members = np.flatnonzero((np.bincount(group)[group] >= 2) & ~new)
        if len(members):
            first = np.flatnonzero(new)[group[members]]
            for a, bb in ((order[first], order[members]), (order[members - 1], order[members])):
                lo, hi = np.minimum(a, bb), np.maximum(a, bb)
                codes.append(lo.astype(np.int64) * n + hi)
    keeper = np.arange(n)
    if codes:
        codes = np.unique(np.concatenate(codes))
        a, bb = codes // n, codes % n
        est = (sig[a] == sig[bb]).mean(axis=1)
        a, bb = a[est >= threshold], bb[est >= threshold]
        if len(a):
            graph = coo_matrix((np.ones(len(a)), (a, bb)), shape=(n, n))
            _, labels = connected_components(graph, directed=False)
            first = np.full(labels.max() + 1, n)
            np.minimum.at(first, labels, np.arange(n))
            keeper = first[labels]
    return keeper, (sig == sig[keeper]).mean(axis=1)


def exact_keys(path):
    df = pd.read_parquet(path, columns=["pair_id", "q_norm", "p_norm"])
    df = df[JOB["alive"][df["pair_id"].to_numpy()]]
    keys = [stable_hash(q + chr(31) + p) for q, p in zip(df["q_norm"], df["p_norm"])]
    return df["pair_id"].to_numpy(), np.array(keys, dtype=np.int64)


def minhash_shard(path):
    df = pd.read_parquet(path, columns=["pair_id", "p_norm"])
    df = df[JOB["alive"][df["pair_id"].to_numpy()]]
    return df["pair_id"].to_numpy(), JOB["minhasher"].signatures(df["p_norm"].tolist())


def run_dedupe(stage):
    if (stats := step_done(stage, "dedupe")) is not None:
        return stats
    f, workers = stage.cfg.filters, stage.cfg.data.workers
    names = stage.source_names
    minhasher = make_minhasher(stage)
    t0, per_source = time.time(), {}
    alive = {s: load_alive(stage, "near_identical", s) for s in names}
    ids, keys, owner = [], [], []
    for i, source in enumerate(names):
        JOB.clear()
        JOB.update(source=source, alive=alive[source])
        res = parallel_map(exact_keys, clean_shards(stage, source), workers, desc=f"exact keys {source}")
        pid = np.concatenate([np.zeros(0, np.int64), *[r[0] for r in res]])
        key = np.concatenate([np.zeros(0, np.int64), *[r[1] for r in res]])
        order = np.argsort(pid)
        ids.append(pid[order])
        keys.append(key[order])
        owner.append(np.full(len(pid), i))
    ids, keys, owner = np.concatenate(ids), np.concatenate(keys), np.concatenate(owner)
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    dup = np.arange(len(keys)) != first[inverse.ravel()]
    for i, source in enumerate(names):
        JOB.clear()
        JOB.update(source=source, minhasher=minhasher)
        mine = dup & (owner == i)
        orig = first[inverse.ravel()[mine]]
        exact = pd.DataFrame({"pair_id": ids[mine], "reason": "exact duplicate (query + passage)",
                              "detail": [f"same text as {names[owner[o]]} #{ids[o]}" for o in orig]})
        still = alive[source].copy()
        still[exact["pair_id"].to_numpy()] = False
        JOB["alive"] = still
        res = parallel_map(minhash_shard, clean_shards(stage, source), workers, desc=f"minhash {source}")
        pid = np.concatenate([np.zeros(0, np.int64), *[r[0] for r in res]])
        sig = np.concatenate([np.zeros((0, f.minhash_perms), np.uint32), *[r[1] for r in res]])
        order = np.argsort(pid)
        pid, sig = pid[order], sig[order]
        keeper, est = near_duplicate_keepers(sig, f.minhash_bands, f.near_dup_jaccard)
        near_rows = np.flatnonzero(keeper != np.arange(len(pid)))
        near = pd.DataFrame({"pair_id": pid[near_rows], "reason": "near-duplicate passage (MinHash)",
                             "detail": [f"of #{pid[keeper[r]]}, est. Jaccard {est[r]:.2f}" for r in near_rows]})
        removed = pd.concat([exact, near], ignore_index=True)
        sample = take_sample(removed)
        if len(sample):
            kept_ids = [int(d.split("#")[1].split(",")[0]) if d.startswith("of #") else None for d in sample["detail"]]
            kept = fetch_texts(stage, source, [k for k in kept_ids if k is not None],
                               ("passage",)).set_index("pair_id")["passage"]
            sample["detail"] = [d + (f" | kept passage: {short(kept[k], 110)}" if k is not None else "")
                                for d, k in zip(sample["detail"], kept_ids)]
        per_source[source] = finish_source(stage, "dedupe", source, alive[source], removed, [sample],
                                           {"near-duplicate groups": int(len(np.unique(keeper[near_rows])))})
    return save_step(stage, "dedupe", per_source, time.time() - t0)
