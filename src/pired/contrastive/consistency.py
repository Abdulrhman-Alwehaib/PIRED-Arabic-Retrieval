import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..common.device import autocast_ctx, exact_fp32, free_gpu
from ..common.io import write_parquet
from ..common.runlog import log
from ..encoder.embedding import pad_chunks
from .steps import JOB, clean_shards, finish_source, load_alive, save_step, step_done, take_sample


@torch.no_grad()
def bge_encode(model, tok, texts, max_len, batch_tokens, device):
    out = torch.empty((len(texts), model.config.hidden_size), dtype=torch.float16, device=device)
    block, n_tokens = 16_384, 0
    with ThreadPoolExecutor(1) as pool:
        def submit(a):
            return pool.submit(tok, texts[a: a + block], truncation=True, max_length=max_len)

        future = submit(0)
        for a in range(0, len(texts), block):
            ids = future.result()["input_ids"]
            if a + block < len(texts):
                future = submit(a + block)
            n_tokens += sum(map(len, ids))
            for rows, x, m in pad_chunks(ids, batch_tokens, tok.pad_token_id):
                x = x.to(device)
                m = torch.ones_like(x) if m is None else m.to(device)
                with autocast_ctx(device):
                    h = model(input_ids=x, attention_mask=m).last_hidden_state[:, 0]
                out[(rows + a).to(device)] = F.normalize(h.float(), dim=-1).half()
    return out, n_tokens


def rank_true_passage(q, p, n_neg, seed, device):
    n = len(q)
    k = min(n_neg, n - 1)
    if k < 1:
        return np.ones(n, dtype=np.int32), (q.float() * p.float()).sum(-1).cpu().numpy(), np.full(n, -1.0)
    rng = np.random.default_rng(seed)
    neg = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        c = rng.choice(n - 1, size=k, replace=False)
        neg[i] = c + (c >= i)
    neg = torch.from_numpy(neg).to(device)
    ranks = torch.empty(n, dtype=torch.int32, device=device)
    s_pos = torch.empty(n, device=device)
    best = torch.empty(n, device=device)
    with exact_fp32():
        for a in range(0, n, 1024):
            b = min(a + 1024, n)
            qa = q[a:b].float()
            s_pos[a:b] = (qa * p[a:b].float()).sum(-1)
            s_neg = torch.bmm(p[neg[a:b]].float(), qa.unsqueeze(-1)).squeeze(-1)
            ranks[a:b] = 1 + (s_neg > s_pos[a:b, None]).sum(1)
            best[a:b] = s_neg.max(1).values
    return ranks.cpu().numpy(), s_pos.cpu().numpy(), best.cpu().numpy()


def bge_groups(stage, source):
    shards = clean_shards(stage, source)
    g = max(1, stage.cfg.filters.bge_shard_rows // stage.cfg.data.shard_rows)
    return [shards[i: i + g] for i in range(0, len(shards), g)]


def rank_bucket(r):
    return "1" if r == 1 else "2" if r == 2 else "3" if r == 3 else "4-10" if r <= 10 else "11-100"


def run_consistency(stage):
    if (stats := step_done(stage, "consistency")) is not None:
        return stats
    f, device, announce = stage.cfg.filters, stage.device, stage.announce
    bge_dir = stage.work / "bge_m3"
    t0, per_source, bge, bge_tok = time.time(), {}, None, None
    gpu = {"seconds": 0.0, "tokens": 0, "pairs": 0}
    keep_top = f.bge_keep_top
    plan = [(si, source, bge_groups(stage, source)) for si, source in enumerate(stage.source_names)]
    total = sum(len(groups) for _, _, groups in plan)
    done, new_shards, new_seconds = 0, 0, 0.0
    try:
        for si, source, groups in plan:
            alive = load_alive(stage, "decontam", source)
            for k, group in enumerate(groups):
                out = bge_dir / source / f"shard-{k:05d}.parquet"
                ids = pd.concat([pd.read_parquet(p, columns=["pair_id"]) for p in group])["pair_id"].to_numpy()
                ids = ids[alive[ids]]
                if out.exists() and np.array_equal(pd.read_parquet(out, columns=["pair_id"])["pair_id"].to_numpy(), ids):
                    done += 1
                    continue
                if bge is None:
                    if device.type != "cuda":
                        raise RuntimeError("D8 (bge-m3 over millions of pairs) needs a CUDA GPU")
                    log(f"loading {f.bge_model}")
                    bge_tok = AutoTokenizer.from_pretrained(f.bge_model)
                    bge = AutoModel.from_pretrained(f.bge_model, dtype=torch.bfloat16,
                                                    use_safetensors=False).to(device).eval()
                    announce(f"bge-m3 {'resumed' if done else 'started'}: {done}/{total} shards already done, "
                             f"next: {source} shard {k + 1}/{len(groups)}")
                df = pd.concat([pd.read_parquet(p, columns=["pair_id", "query", "passage"]) for p in group])
                df = df[alive[df["pair_id"].to_numpy()]].reset_index(drop=True)
                g0 = time.time()
                q, nq = bge_encode(bge, bge_tok, df["query"].tolist(), f.bge_query_max_len, f.bge_batch_tokens, device)
                p, npass = bge_encode(bge, bge_tok, df["passage"].tolist(), f.bge_passage_max_len, f.bge_batch_tokens,
                                      device)
                ranks, score, best = rank_true_passage(q, p, f.bge_negatives, [stage.cfg.data.seed, si, k], device)
                torch.cuda.synchronize()
                write_parquet(pd.DataFrame({"pair_id": df["pair_id"].to_numpy(), "rank": ranks.astype(np.int16),
                                            "score": score, "best_negative": best}), out)
                seconds = time.time() - g0
                gpu["seconds"] += seconds
                gpu["tokens"] += nq + npass
                gpu["pairs"] += len(df)
                done, new_shards, new_seconds = done + 1, new_shards + 1, new_seconds + seconds
                eta_h = (total - done) * new_seconds / new_shards / 3600
                announce(f"bge-m3 {source} shard {k + 1}/{len(groups)} done: {len(df):,} pairs in {seconds:.0f}s "
                         f"({(nq + npass) / seconds:,.0f} tokens/s), kept {float((ranks <= keep_top).mean()):.1%} | "
                         f"all sources {done}/{total} shards, ETA ~{eta_h:.1f} h")
                del q, p
    except BaseException as e:
        announce(f"bge-m3 stopped ({type(e).__name__}) with {done}/{total} shards done; run D8 again to resume")
        raise
    if bge is not None:
        del bge
        free_gpu()
    announce(f"bge-m3: all {total} shards scored, applying the top-{keep_top} cutoff")
    for source in stage.source_names:
        JOB.clear()
        JOB["source"] = source
        alive = load_alive(stage, "decontam", source)
        scores = pd.concat([pd.read_parquet(p) for p in sorted((bge_dir / source).glob("shard-*.parquet"))],
                           ignore_index=True)
        scores = scores[scores["pair_id"].isin(np.flatnonzero(alive))]
        if len(scores) != alive.sum():
            raise AssertionError(f"{source}: {len(scores)} scores for {alive.sum()} pairs")
        bad = scores[scores["rank"] > keep_top]
        removed = pd.DataFrame({"pair_id": bad["pair_id"].to_numpy(),
                                "reason": f"bge-m3: true passage ranked below {keep_top} of {f.bge_negatives + 1}",
                                "detail": [f"rank {r}, score {s:.3f}, best negative {b:.3f}"
                                           for r, s, b in zip(bad["rank"], bad["score"], bad["best_negative"])]})
        hist = Counter(rank_bucket(int(r)) for r in scores["rank"])
        info = {"rank histogram": {b: hist.get(b, 0) for b in ("1", "2", "3", "4-10", "11-100")},
                "kept at top-1/3/5/10": [round(float((scores["rank"] <= t).mean()), 4) for t in (1, 3, 5, 10)],
                "median score": round(float(scores["score"].median()), 4) if len(scores) else None}
        per_source[source] = finish_source(stage, "consistency", source, alive, removed, [take_sample(removed)], info)
    stats = save_step(stage, "consistency", per_source, time.time() - t0)
    if gpu["pairs"]:
        stats["gpu"] = gpu
        (stage.steps_dir / "consistency" / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                                                                    encoding="utf-8")
    kept, seen = sum(s["out"] for s in per_source.values()), sum(s["in"] for s in per_source.values())
    announce(f"D8 finished: kept {kept:,} of {seen:,} pairs ({kept / max(seen, 1):.1%})")
    return stats


def format_consistency(stage, stats):
    lines = ["rank of the true passage among 100 (share of pairs kept if the cutoff were top-1 / 3 / 5 / 10):"]
    for source, s in stats["sources"].items():
        lines.append(f"  {source:<10} {s['rank histogram']}  kept at top-1/3/5/10: {s['kept at top-1/3/5/10']}  "
                     f"median score {s['median score']}")
    if "gpu" in stats:
        g = stats["gpu"]
        stage.timings["consistency_gpu"] = g
        lines.append(f"bge-m3 on this GPU: {g['pairs']:,} pairs, {g['tokens']:,} XLM-R tokens in {g['seconds']:.0f}s = "
                     f"{g['tokens'] / g['seconds']:,.0f} tokens/s, {g['tokens'] / g['pairs']:.0f} tokens per pair")
    return "\n".join(lines)
