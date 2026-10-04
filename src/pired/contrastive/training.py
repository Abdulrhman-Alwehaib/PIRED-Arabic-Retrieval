import contextlib
import dataclasses
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from ..common.checkpoints import RandState, read_training_state, restore_rng_states, rng_states, write_checkpoint
from ..common.device import autocast_ctx, exact_fp32
from ..common.runlog import log
from ..encoder.embedding import pad_chunks
from ..encoder.weights import load_weights_strict
from .mix import final_dir, final_split


def ensure_final_data(stage):
    final = final_dir(stage)
    if (final / "manifest.json").exists():
        return
    repo = stage.repo_id("data")
    log(f"{final} not found: downloading {repo}")
    snapshot_download(repo, repo_type="dataset", local_dir=final, token=stage.token)


class PairData:
    def __init__(self, stage, split):
        import datasets as hf_datasets

        hf_datasets.disable_progress_bars()
        self.tables = {}
        for s in stage.source_names:
            files = sorted(str(p) for p in (final_dir(stage) / split / s).glob("*.parquet"))
            if files:
                self.tables[s] = hf_datasets.Dataset.from_parquet(files, cache_dir=str(stage.work / "hf_cache"))
        self.sizes = {s: len(t) for s, t in self.tables.items()}

    def rows(self, source, idx):
        batch = self.tables[source][[int(i) for i in idx]]
        return batch["query"], batch["passage"]


def ids_hash(ids):
    return int.from_bytes(hashlib.blake2b(np.asarray(ids, dtype=np.uint16).tobytes(), digest_size=8).digest(),
                          "little", signed=True)


def content_hashes(id_lists, n_prefix):
    return torch.tensor([ids_hash(ids[1 + n_prefix: -1]) for ids in id_lists], dtype=torch.int64)


@dataclass
class Batch:
    source: str
    size: int
    q_chunks: list
    p_chunks: list
    q_hash: torch.Tensor
    p_hash: torch.Tensor
    n_tokens: int


def make_batch(stage, queries, passages, source, chunk_tokens):
    embedder = stage.embedder
    q_ids, p_ids = embedder.tokenize(queries, "query"), embedder.tokenize(passages, "passage")
    return Batch(source, len(queries), pad_chunks(q_ids, chunk_tokens, stage.pad_id),
                 pad_chunks(p_ids, chunk_tokens, stage.pad_id), content_hashes(q_ids, stage.n_q_prefix),
                 content_hashes(p_ids, stage.n_p_prefix), sum(map(len, q_ids)) + sum(map(len, p_ids)))


class BatchLoader:
    def __init__(self, stage, data, batch_size, seed, chunk_tokens):
        self.stage, self.data, self.batch_size, self.seed, self.chunk_tokens = stage, data, batch_size, seed, chunk_tokens
        labels = [s for s in stage.source_names for _ in range(data.sizes.get(s, 0) // batch_size)]
        random.Random(seed).shuffle(labels)
        seen, self.schedule = Counter(), []
        for s in labels:
            self.schedule.append((s, seen[s]))
            seen[s] += 1
        self.perms = {s: np.random.default_rng([seed, stage.source_names.index(s)]).permutation(n)
                      for s, n in data.sizes.items()}
        self.pool, self.pending = ThreadPoolExecutor(1), {}

    def __len__(self):
        return len(self.schedule)

    def fingerprint(self):
        return {"sizes": self.data.sizes, "batch_size": self.batch_size, "seed": self.seed, "steps": len(self)}

    def batches_per_source(self):
        return dict(Counter(s for s, _ in self.schedule))

    def _build(self, step):
        source, j = self.schedule[step]
        rows = self.perms[source][j * self.batch_size: (j + 1) * self.batch_size]
        return make_batch(self.stage, *self.data.rows(source, rows), source, self.chunk_tokens)

    def get(self, step):
        future = self.pending.pop(step, None) or self.pool.submit(self._build, step)
        self.pending = {}
        if step + 1 < len(self):
            self.pending[step + 1] = self.pool.submit(self._build, step + 1)
        return future.result()


def contrastive_loss(q, p, q_hash, p_hash, temperature, bidirectional):
    n = q.shape[0]
    labels = torch.arange(n, device=q.device)
    eye = torch.eye(n, dtype=torch.bool, device=q.device)
    sim = q @ p.T
    mask_qp = ((p_hash[None, :] == p_hash[:, None]) | (p_hash[None, :] == q_hash[:, None])) & ~eye
    logits_qp = (sim / temperature).masked_fill(mask_qp, float("-inf"))
    loss = F.cross_entropy(logits_qp, labels)
    stats = {"loss_qp": loss.detach(), "acc_qp": (logits_qp.argmax(1) == labels).float().mean(),
             "masked": mask_qp.sum()}
    if bidirectional:
        mask_pq = ((q_hash[None, :] == q_hash[:, None]) | (q_hash[None, :] == p_hash[:, None])) & ~eye
        logits_pq = (sim.T / temperature).masked_fill(mask_pq, float("-inf"))
        loss_pq = F.cross_entropy(logits_pq, labels)
        stats.update(loss_pq=loss_pq.detach(), acc_pq=(logits_pq.argmax(1) == labels).float().mean(),
                     masked=stats["masked"] + mask_pq.sum())
        loss = (loss + loss_pq) / 2
    return loss, stats


def embed_chunk(model, ids, mask, use_autocast, device):
    ids = ids.to(device, non_blocking=True)
    mask = None if mask is None else mask.to(device, non_blocking=True)
    with autocast_ctx(device) if use_autocast else contextlib.nullcontext():
        return model.embed(ids, mask)


def batch_loss(q, p, batch, s, device):
    with exact_fp32(), torch.autocast(device.type, enabled=False):
        return contrastive_loss(q.float(), p.float(), batch.q_hash.to(device), batch.p_hash.to(device),
                                s.temperature, s.bidirectional)


def grad_cache_step(model, batch, s, device, use_autocast=True):
    reps, states = {}, {}
    with torch.no_grad():
        for side, chunks in (("q", batch.q_chunks), ("p", batch.p_chunks)):
            out = torch.empty((batch.size, model.config.hidden_size), dtype=torch.float32, device=device)
            for c, (rows, ids, mask) in enumerate(chunks):
                states[side, c] = RandState(device)
                out[rows.to(device)] = embed_chunk(model, ids, mask, use_autocast, device).float()
            reps[side] = out.requires_grad_()
    with exact_fp32(), torch.autocast(device.type, enabled=False):
        loss, stats = contrastive_loss(reps["q"], reps["p"], batch.q_hash.to(device), batch.p_hash.to(device),
                                       s.temperature, s.bidirectional)
        loss.backward()
    for side, chunks in (("q", batch.q_chunks), ("p", batch.p_chunks)):
        grad = reps[side].grad
        for c, (rows, ids, mask) in enumerate(chunks):
            with states[side, c].replay():
                emb = embed_chunk(model, ids, mask, use_autocast, device)
            emb.float().backward(grad[rows.to(device)])
    return loss.detach(), stats


def full_batch_step(model, batch, s, device, use_autocast=True):
    reps = {}
    for side, chunks in (("q", batch.q_chunks), ("p", batch.p_chunks)):
        embs = [embed_chunk(model, ids, mask, use_autocast, device).float() for _, ids, mask in chunks]
        rows = torch.cat([r for r, _, _ in chunks]).to(device)
        reps[side] = torch.cat(embs)[torch.argsort(rows)]
    loss, stats = batch_loss(reps["q"], reps["p"], batch, s, device)
    with exact_fp32():
        loss.backward()
    return loss.detach(), stats


class ValidationSet:
    def __init__(self, stage):
        self.df = pd.concat([final_split(stage, "validation", s) for s in stage.source_names], ignore_index=True)
        self.q_ids = stage.embedder.tokenize(self.df["query"].tolist(), "query")
        self.p_ids = stage.embedder.tokenize(self.df["passage"].tolist(), "passage")
        self.p_hash = content_hashes(self.p_ids, stage.n_p_prefix)
        self.source = self.df["source"].to_numpy()
        self.source_names = stage.source_names

    def __len__(self):
        return len(self.df)


def ranking_metrics(ranks):
    r = ranks.float()
    return {"ndcg@10": float(torch.where(r <= 10, 1.0 / torch.log2(r + 1), torch.zeros_like(r)).mean()),
            "recall@10": float((r <= 10).float().mean()), "recall@100": float((r <= 100).float().mean()),
            "mrr@10": float(torch.where(r <= 10, 1.0 / r, torch.zeros_like(r)).mean())}


def pair_metrics(q, p, val, n_random, device):
    n, p_hash = len(q), val.p_hash.to(device)
    ranks = torch.empty(n, dtype=torch.long, device=device)
    with exact_fp32():
        for a in range(0, n, 1024):
            b = min(a + 1024, n)
            s = q[a:b] @ p.T
            pos = s[torch.arange(b - a, device=device), torch.arange(a, b, device=device)]
            ranks[a:b] = 1 + ((s > pos[:, None]) & (p_hash[None, :] != p_hash[a:b, None])).sum(1)
    out = ranking_metrics(ranks)
    out["per_source"] = {s: ranking_metrics(ranks[torch.from_numpy(val.source == s).to(device)])
                         for s in val.source_names if (val.source == s).any()}
    g = torch.Generator().manual_seed(0)
    i, j = torch.randint(n, (n_random,), generator=g), torch.randint(n, (n_random,), generator=g)
    i, j = i[i != j].to(device), j[i != j].to(device)
    out["cos_pairs"] = float((q * p).sum(-1).mean())
    out["cos_unrelated_qp"] = float((q[i] * p[j]).sum(-1).mean())
    out["cos_unrelated_pp"] = float((p[i] * p[j]).sum(-1).mean())
    return out


@torch.no_grad()
def evaluate_pairs(stage, model, val, batch_tokens, n_random):
    embedder = stage.make_embedder(model)
    q = embedder.embed_ids(val.q_ids, batch_tokens)
    p = embedder.embed_ids(val.p_ids, batch_tokens)
    return pair_metrics(q, p, val, n_random, stage.device)


def val_summary(m):
    return (f"nDCG@10 {m['ndcg@10']:.4f}  R@10 {m['recall@10']:.4f}  R@100 {m['recall@100']:.4f}  | cos: pairs "
            f"{m['cos_pairs']:.3f}, unrelated q-p {m['cos_unrelated_qp']:.3f}, p-p {m['cos_unrelated_pp']:.3f}")


@dataclass
class TrainState:
    step: int = 0
    pairs_seen: int = 0
    tokens_seen: int = 0
    elapsed_seconds: float = 0.0


def build_optimizer(model, s, device):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": s.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=s.lr, betas=s.betas, eps=s.eps, fused=device.type == "cuda")


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def build_scheduler(optimizer, total, s):
    warmup = max(1, round(s.warmup_fraction * total))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda k: lr_factor(k, total, warmup))


def save_checkpoint(root, model, optimizer, scheduler, state, loader, settings):
    return write_checkpoint(root, state.step, model, {
        "state": dataclasses.asdict(state), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "data": {"position": state.step, **loader.fingerprint()},
        "rng": rng_states(), "train_settings": dataclasses.asdict(settings)})


def load_checkpoint(ckpt, model, optimizer, scheduler, loader):
    load_weights_strict(model, ckpt)
    ts = read_training_state(ckpt)
    optimizer.load_state_dict(ts["optimizer"])
    scheduler.load_state_dict(ts["scheduler"])
    saved = {k: v for k, v in ts["data"].items() if k != "position"}
    if saved != json.loads(json.dumps(loader.fingerprint())):
        raise ValueError(f"the training data or batch settings changed since this checkpoint: {saved} vs "
                         f"{loader.fingerprint()}")
    restore_rng_states(ts["rng"])
    return TrainState(**ts["state"])


def train_steps(stage, model, optimizer, scheduler, loader, state, *, total_steps, stop_at=None, logger=None,
                val=None, on_checkpoint=None, record_losses=False, use_autocast=True):
    s, device, end = stage.cfg.train, stage.device, min(stop_at or total_steps, total_steps)
    model.train()
    losses, window, n_window, tokens_window, previous_cos = [], defaultdict(float), 0, 0, None
    t_window = time.perf_counter()
    while state.step < end:
        batch = loader.get(state.step)
        optimizer.zero_grad(set_to_none=True)
        loss, st = grad_cache_step(model, batch, s, device, use_autocast)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), s.max_grad_norm)
        optimizer.step()
        scheduler.step()
        state.step += 1
        state.pairs_seen += batch.size
        state.tokens_seen += batch.n_tokens
        tokens_window += batch.n_tokens
        n_window += 1
        for k, v in {"loss": loss, **st, "grad_norm": grad_norm}.items():
            window[k] += float(v)
        if record_losses:
            losses.append(float(loss))
        if logger is not None and (state.step % s.log_every == 0 or state.step == end):
            now = time.perf_counter()
            state.elapsed_seconds += now - t_window
            logger.log({"step": state.step, "source": batch.source, **{k: v / n_window for k, v in window.items()},
                        "lr": scheduler.get_last_lr()[0], "tokens_per_sec": tokens_window / (now - t_window),
                        "pairs_seen": state.pairs_seen, "elapsed_h": state.elapsed_seconds / 3600,
                        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0})
            window, n_window, tokens_window, t_window = defaultdict(float), 0, 0, time.perf_counter()
        if val is not None and (state.step % s.eval_every == 0 or state.step == total_steps):
            paused = time.perf_counter()
            m = evaluate_pairs(stage, model, val, s.eval_batch_tokens, s.val_random_pairs)
            model.train()
            cos = max(m["cos_unrelated_qp"], m["cos_unrelated_pp"])
            if logger is not None:
                logger.log({"step": state.step, "event": "validation",
                            **{k: v for k, v in m.items() if k != "per_source"},
                            "per_source_ndcg@10": {k: round(v["ndcg@10"], 4) for k, v in m["per_source"].items()}})
                rising = cos >= s.collapse_watch and previous_cos is not None and cos > previous_cos
                if cos >= s.collapse_cosine or rising:
                    shown = previous_cos if previous_cos is None else round(previous_cos, 3)
                    logger.log({"step": state.step, "event": "WARNING collapse",
                                "detail": f"mean cosine of unrelated texts is {cos:.3f} (previous validation: "
                                          f"{shown}); embeddings may be collapsing"})
            previous_cos = cos
            t_window += time.perf_counter() - paused
        if on_checkpoint is not None and state.step % s.save_every == 0:
            paused = time.perf_counter()
            on_checkpoint(state)
            t_window += time.perf_counter() - paused
    return losses
