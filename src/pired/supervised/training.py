import dataclasses
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, snapshot_download

from ..common.checkpoints import (RandState, list_hub_checkpoints, local_checkpoints, read_training_state,
                                  restore_rng_states, rng_states)
from ..common.device import exact_fp32
from ..common.io import replace_dir, write_json
from ..common.runlog import JsonlLogger
from ..contrastive.training import build_optimizer, build_scheduler, content_hashes, embed_chunk
from ..encoder.embedding import pad_chunks
from ..encoder.weights import load_weights_strict

__all__ = ["Batch", "BatchLoader", "EventLogger", "TrainData", "TrainState", "ValidationSet", "batch_loss",
           "build_optimizer", "build_scheduler", "contrastive_loss", "evaluate_retrieval", "find_resume_checkpoint",
           "full_batch_step", "grad_cache_step", "load_checkpoint", "make_batch", "retrieval_metrics",
           "save_best", "save_checkpoint", "train_steps", "val_summary"]


class TrainData:
    def __init__(self, stage, folder):
        import datasets as hf_datasets

        hf_datasets.disable_progress_bars()
        self.tables = {}
        for s in stage.source_names:
            files = sorted(str(p) for p in (Path(folder) / "train" / s).glob("*.parquet"))
            if files:
                self.tables[s] = hf_datasets.Dataset.from_parquet(
                    files, cache_dir=str(stage.work / "hf_cache"),
                    columns=["query", "positives", "negatives", "neg_scores"])
        self.sizes = {s: len(t) for s, t in self.tables.items()}

    def rows(self, source, idx):
        b = self.tables[source][[int(i) for i in idx]]
        return b["query"], b["positives"], b["negatives"], b["neg_scores"]


@dataclass
class Batch:
    source: str
    size: int
    q_chunks: list
    p_chunks: list
    q_hash: torch.Tensor
    p_hash: torch.Tensor
    pos_hash: torch.Tensor
    n_tokens: int
    skip: torch.Tensor = field(default_factory=lambda: torch.zeros((0, 2), dtype=torch.long))


def make_batch(stage, n_neg, queries, positives, negatives, neg_scores, source, chunk_tokens):
    per = 1 + n_neg
    passages = [t for pos, neg in zip(positives, negatives) for t in [pos[0], *list(neg)[:n_neg]]]
    if len(passages) != per * len(queries):
        raise AssertionError("every row needs exactly one positive and n_neg negatives")
    q_ids, p_ids = stage.tokenize_texts(queries, "query"), stage.tokenize_texts(passages, "passage")
    extra = [list(pos[1:]) for pos in positives]
    extra_ids = stage.tokenize_texts([t for e in extra for t in e], "passage") if any(extra) else []
    p_hash = content_hashes(p_ids, stage.n_p_prefix).tolist()
    e_hash, k = content_hashes(extra_ids, stage.n_p_prefix).tolist() if extra_ids else [], 0
    width = 1 + max(len(e) for e in extra)
    pos_hash = torch.full((len(queries), width), np.iinfo(np.int64).min, dtype=torch.int64)
    for i, e in enumerate(extra):
        pos_hash[i, 0] = p_hash[per * i]
        pos_hash[i, 1: 1 + len(e)] = torch.tensor(e_hash[k: k + len(e)], dtype=torch.int64)
        k += len(e)
    top = stage.cfg.train.max_neg_score
    skip = [(i, per * i + 1 + j) for i, scores in enumerate(neg_scores) for j, s in enumerate(list(scores)[:n_neg])
            if top is not None and s is not None and s == s and s > top]
    return Batch(source, len(queries), pad_chunks(q_ids, chunk_tokens, stage.pad_id),
                 pad_chunks(p_ids, chunk_tokens, stage.pad_id), content_hashes(q_ids, stage.n_q_prefix),
                 torch.tensor(p_hash), pos_hash, sum(map(len, q_ids)) + sum(map(len, p_ids)),
                 torch.tensor(skip, dtype=torch.long).reshape(-1, 2))


class BatchLoader:
    def __init__(self, stage, data, n_neg, batch_size, seed, chunk_tokens):
        self.stage, self.data, self.n_neg = stage, data, n_neg
        self.batch_size, self.seed, self.chunk_tokens = batch_size, seed, chunk_tokens
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

    def rows_batch(self, source, rows, chunk_tokens):
        return make_batch(self.stage, self.n_neg, *self.data.rows(source, rows), source, chunk_tokens)

    def _build(self, step):
        source, j = self.schedule[step]
        rows = self.perms[source][j * self.batch_size: (j + 1) * self.batch_size]
        return self.rows_batch(source, rows, self.chunk_tokens)

    def get(self, step):
        future = self.pending.pop(step, None) or self.pool.submit(self._build, step)
        self.pending = {}
        if step + 1 < len(self):
            self.pending[step + 1] = self.pool.submit(self._build, step + 1)
        return future.result()


def contrastive_loss(q, p, batch, temperature, bidirectional):
    n = q.shape[0]
    per = p.shape[0] // n
    dev = q.device
    labels = torch.arange(n, device=dev) * per
    p_hash, q_hash, pos_hash = batch.p_hash.to(dev), batch.q_hash.to(dev), batch.pos_hash.to(dev)
    own = torch.zeros((n, p.shape[0]), dtype=torch.bool, device=dev)
    own[torch.arange(n, device=dev), labels] = True
    same_as_pos = (p_hash[None, None, :] == pos_hash[:, :, None]).any(1)
    mask = (same_as_pos | (p_hash[None, :] == q_hash[:, None])) & ~own
    if batch.skip.numel():
        mask[batch.skip[:, 0].to(dev), batch.skip[:, 1].to(dev)] = True
    sim = q @ p.T
    logits = (sim / temperature).masked_fill(mask, float("-inf"))
    loss = F.cross_entropy(logits, labels)
    stats = {"loss_qp": loss.detach(), "acc": (logits.argmax(1) == labels).float().mean(), "masked": mask.sum(),
             "skipped_negs": batch.skip.shape[0]}
    if bidirectional:
        sim_pq = sim[:, labels].T
        eye = torch.eye(n, dtype=torch.bool, device=dev)
        mask_pq = (p_hash[labels][:, None, None] == pos_hash[None, :, :]).any(2) & ~eye
        logits_pq = (sim_pq / temperature).masked_fill(mask_pq, float("-inf"))
        loss_pq = F.cross_entropy(logits_pq, torch.arange(n, device=dev))
        stats.update(loss_pq=loss_pq.detach(), masked=stats["masked"] + mask_pq.sum())
        loss = (loss + loss_pq) / 2
    return loss, stats


def batch_loss(q, p, batch, s, device):
    with exact_fp32(), torch.autocast(device.type, enabled=False):
        return contrastive_loss(q.float(), p.float(), batch, s.temperature, s.bidirectional)


def grad_cache_step(model, batch, s, device, use_autocast=True):
    reps, states = {}, {}
    n_passages = sum(len(rows) for rows, _, _ in batch.p_chunks)
    sizes = {"q": batch.size, "p": n_passages}
    with torch.no_grad():
        for side, chunks in (("q", batch.q_chunks), ("p", batch.p_chunks)):
            out = torch.empty((sizes[side], model.config.hidden_size), dtype=torch.float32, device=device)
            for c, (rows, ids, mask) in enumerate(chunks):
                states[side, c] = RandState(device)
                out[rows.to(device)] = embed_chunk(model, ids, mask, use_autocast, device).float()
            reps[side] = out.requires_grad_()
    with exact_fp32(), torch.autocast(device.type, enabled=False):
        loss, stats = contrastive_loss(reps["q"], reps["p"], batch, s.temperature, s.bidirectional)
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
    def __init__(self, stage, folder):
        self.q = pd.read_parquet(Path(folder) / "validation" / "queries.parquet")
        corpus = pd.read_parquet(Path(folder) / "validation" / "corpus.parquet").sort_values("cid")
        if not (corpus["cid"].to_numpy() == np.arange(len(corpus))).all():
            raise AssertionError("the validation corpus ids are not 0..n-1")
        self.q_ids = stage.tokenize_texts(self.q["query"].tolist(), "query")
        self.c_ids = stage.tokenize_texts(corpus["text"].tolist(), "passage")
        self.relevant = [set(map(int, p)) for p in self.q["positives"]]
        self.source = self.q["source"].to_numpy()
        self.source_names = stage.source_names
        self.n_corpus = len(corpus)

    def __len__(self):
        return len(self.q)


def retrieval_metrics(top, relevant):
    disc = 1.0 / np.log2(np.arange(2, 12))
    ndcg, r10, r100 = [], [], []
    for row, rel in zip(top, relevant):
        hits = np.array([c in rel for c in row[:100]])
        top10 = hits[:10]
        ndcg.append(float((top10 * disc[: len(top10)]).sum() / disc[: min(len(rel), 10)].sum()))
        r10.append(hits[:10].sum() / len(rel))
        r100.append(hits.sum() / len(rel))
    return {"ndcg@10": float(np.mean(ndcg)), "recall@10": float(np.mean(r10)), "recall@100": float(np.mean(r100))}


@torch.no_grad()
def evaluate_retrieval(stage, model, val, batch_tokens, n_random):
    device = stage.device
    embedder = stage.make_embedder(model)
    qv = embedder.embed_ids(val.q_ids, batch_tokens, dtype=torch.float16)
    cv = embedder.embed_ids(val.c_ids, batch_tokens, dtype=torch.float16)
    k = min(100, val.n_corpus)
    top = np.concatenate([(qv[a: a + 2_048] @ cv.T).float().topk(k, dim=1).indices.cpu().numpy()
                          for a in range(0, len(qv), 2_048)])
    out = retrieval_metrics(top, val.relevant)
    out["per_source"] = {s: retrieval_metrics(top[val.source == s],
                                              [r for r, x in zip(val.relevant, val.source) if x == s])
                         for s in val.source_names if (val.source == s).any()}
    g = torch.Generator().manual_seed(0)
    i, j = torch.randint(len(qv), (n_random,), generator=g), torch.randint(val.n_corpus, (n_random,), generator=g)
    a, b = torch.randint(val.n_corpus, (n_random,), generator=g), torch.randint(val.n_corpus, (n_random,), generator=g)
    out["cos_unrelated_qp"] = float((qv[i.to(device)].float() * cv[j.to(device)].float()).sum(-1).mean())
    out["cos_unrelated_pp"] = float((cv[a.to(device)].float() * cv[b.to(device)].float()).sum(-1).mean())
    return out


def val_summary(m):
    return (f"nDCG@10 {m['ndcg@10']:.4f}  R@10 {m['recall@10']:.4f}  R@100 {m['recall@100']:.4f}  | cos of unrelated "
            f"texts: q-p {m['cos_unrelated_qp']:.3f}, p-p {m['cos_unrelated_pp']:.3f}")


@dataclass
class TrainState:
    step: int = 0
    queries_seen: int = 0
    tokens_seen: int = 0
    elapsed_seconds: float = 0.0
    best_ndcg: float = -1.0
    best_step: int = 0


def save_checkpoint(root, model, optimizer, scheduler, state, loader, settings):
    root = Path(root)
    final, tmp = root / f"step-{state.step:08d}", root / f".tmp-step-{state.step:08d}"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    torch.save({"state": dataclasses.asdict(state), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "data": {"position": state.step, **loader.fingerprint()},
                "rng": rng_states(), "train_settings": dataclasses.asdict(settings)}, tmp / "training_state.pt")
    return replace_dir(tmp, final)


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


def save_best(root, model, state, metrics):
    final, tmp = Path(root) / "best", Path(root) / ".tmp-best"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    write_json({"step": state.step, **{k: v for k, v in metrics.items() if k != "per_source"},
                "per_source": metrics.get("per_source")}, tmp / "metrics.json")
    return replace_dir(tmp, final)


def find_resume_checkpoint(root, repo_id, run_name, token):
    local = local_checkpoints(root)
    local_step = int(local[-1].name.split("-")[1]) if local else -1
    if repo_id:
        remote = list_hub_checkpoints(HfApi(token=token), repo_id, f"checkpoints/{run_name}")
        if remote and int(remote[-1].split("-")[1]) > local_step:
            staging = Path(root) / ".hub_download"
            snapshot_download(repo_id, allow_patterns=[f"checkpoints/{run_name}/{remote[-1]}/*",
                                                       f"checkpoints/{run_name}/best/*"], local_dir=staging,
                              token=token)
            for name in (remote[-1], "best"):
                src = staging / "checkpoints" / run_name / name
                if src.exists():
                    shutil.rmtree(Path(root) / name, ignore_errors=True)
                    shutil.move(str(src), str(Path(root) / name))
            shutil.rmtree(staging, ignore_errors=True)
            return Path(root) / remote[-1]
    return local[-1] if local else None


class EventLogger(JsonlLogger):
    def __init__(self, path, event_notifier=None, echo=True):
        super().__init__(path, echo=echo)
        self.event_notifier = event_notifier

    def log(self, record):
        super().log(record)
        if "event" in record and self.event_notifier is not None:
            line = " | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()
                              if not isinstance(v, dict))
            self.event_notifier.send(f"train {line}")


def train_steps(stage, model, optimizer, scheduler, loader, state, *, total_steps, stop_at=None, logger=None, val=None,
                on_checkpoint=None, on_best=None, record_losses=False, use_autocast=True):
    s, device, end = stage.cfg.train, stage.device, min(stop_at or total_steps, total_steps)
    model.train()
    losses, window, n_window, tokens_window, previous_cos = [], defaultdict(float), 0, 0, None
    t_window = last_save = time.perf_counter()
    while state.step < end:
        batch = loader.get(state.step)
        optimizer.zero_grad(set_to_none=True)
        loss, st = grad_cache_step(model, batch, s, device, use_autocast)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), s.max_grad_norm)
        optimizer.step()
        scheduler.step()
        state.step += 1
        state.queries_seen += batch.size
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
                        "queries_seen": state.queries_seen, "elapsed_h": state.elapsed_seconds / 3600,
                        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0})
            window, n_window, tokens_window, t_window = defaultdict(float), 0, 0, time.perf_counter()
        if val is not None and (state.step % s.eval_every == 0 or state.step == total_steps):
            paused = time.perf_counter()
            m = evaluate_retrieval(stage, model, val, s.eval_batch_tokens, s.random_pairs)
            model.train()
            cos = max(m["cos_unrelated_qp"], m["cos_unrelated_pp"])
            improved = m["ndcg@10"] > state.best_ndcg
            if improved:
                state.best_ndcg, state.best_step = m["ndcg@10"], state.step
                if on_best is not None:
                    on_best(state, m)
            if logger is not None:
                logger.log({"step": state.step, "event": "validation",
                            **{k: v for k, v in m.items() if k != "per_source"}, "best": improved,
                            "per_source_ndcg@10": {k: round(v["ndcg@10"], 4) for k, v in m["per_source"].items()}})
                rising = cos >= s.collapse_watch and previous_cos is not None and cos > previous_cos
                if cos >= s.collapse_cosine or rising:
                    shown = previous_cos if previous_cos is None else round(previous_cos, 3)
                    logger.log({"step": state.step, "event": "WARNING collapse",
                                "detail": f"mean cosine of unrelated texts is {cos:.3f} (previous validation: {shown})"})
            previous_cos = cos
            t_window += time.perf_counter() - paused
            last_save += time.perf_counter() - paused
        due = (s.save_every_steps and state.step % s.save_every_steps == 0) or (
            s.save_every_minutes and time.perf_counter() - last_save >= 60 * s.save_every_minutes)
        if on_checkpoint is not None and due and state.step < end:
            paused = time.perf_counter()
            on_checkpoint(state)
            last_save = time.perf_counter()
            t_window += last_save - paused
    return losses
