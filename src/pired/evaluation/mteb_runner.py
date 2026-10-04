import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download

from ..common.device import autocast_ctx, free_gpu
from ..common.runlog import log
from ..encoder.embedding import pad_chunks
from ..encoder.modeling import mean_pool

warnings.filterwarnings("ignore", message='Field "model_type" has conflict with protected namespace')

import datasets as hf_datasets
import mteb
from mteb.abstasks.retrieval_dataset_loaders import RetrievalSplitData
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import PromptType

hf_datasets.disable_progress_bars()


class RetrievalEncoder(AbsEncoder):
    def __init__(self, name, embed, tokenizer, prefixes, max_len, embed_dim, batch_tokens, device,
                 mask_full_chunks=True, min_batch_tokens=None):
        self.embed, self.tokenizer, self.prefixes, self.max_len = embed, tokenizer, prefixes, max_len
        self.embed_dim, self.batch_tokens, self.device = embed_dim, batch_tokens, device
        self.mask_full_chunks, self.min_batch_tokens = mask_full_chunks, min_batch_tokens
        self.mteb_model_meta = ModelMeta.create_empty(overwrites={
            "name": name, "revision": "local", "similarity_fn_name": ScoringFunction.COSINE, "embed_dim": embed_dim,
            "framework": ["PyTorch"], "max_tokens": max_len[1]})
        self.chars, self.tokens, self.seconds = 0, 0, 0.0

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        texts = [t for batch in inputs for t in batch["text"]]
        return self.encode_texts(texts, is_query=prompt_type == PromptType.query)

    @torch.no_grad()
    def encode_texts(self, texts, is_query):
        t0 = time.time()
        prefix, max_len = (self.prefixes[0], self.max_len[0]) if is_query else (self.prefixes[1], self.max_len[1])
        ids = self.tokenizer([prefix + t for t in texts], truncation=True, max_length=max_len)["input_ids"]
        while True:
            try:
                out = self.embed_ids(ids)
                break
            except torch.cuda.OutOfMemoryError:
                free_gpu()
                if self.min_batch_tokens is None or self.batch_tokens // 2 < self.min_batch_tokens:
                    raise
                self.batch_tokens //= 2
                log(f"    out of GPU memory: batch lowered to {self.batch_tokens:,} tokens, trying again")
        self.chars += sum(map(len, texts))
        self.tokens += sum(map(len, ids))
        self.seconds += time.time() - t0
        return out

    def embed_ids(self, ids):
        device = self.device
        out = torch.empty((len(ids), self.embed_dim), dtype=torch.float32, device=device)
        for rows, x, m in pad_chunks(ids, self.batch_tokens, self.tokenizer.pad_token_id, self.mask_full_chunks):
            with autocast_ctx(device):
                out[rows.to(device)] = self.embed(x.to(device), None if m is None else m.to(device)).float()
        return out.cpu().numpy()


def ours_encoder(model, tokenizer, name, prefixes, max_len, batch_tokens, device, mask_full_chunks=True,
                 min_batch_tokens=None):
    model.eval()
    return RetrievalEncoder(name, model.embed, tokenizer, prefixes, max_len, model.config.hidden_size, batch_tokens,
                            device, mask_full_chunks, min_batch_tokens)


def baseline_encoder(name, prefixes, max_len, batch_tokens, device, mask_full_chunks=True, min_batch_tokens=None):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()

    def embed(x, mask):
        attention = torch.ones_like(x) if mask is None else mask
        hidden = model(input_ids=x, attention_mask=attention).last_hidden_state
        return F.normalize(mean_pool(hidden.float(), mask), p=2, dim=-1)

    encoder = RetrievalEncoder(name, embed, tokenizer, prefixes, (max_len, max_len), model.config.hidden_size,
                               batch_tokens, device, mask_full_chunks, min_batch_tokens)
    return encoder, model


def smoke_subset(task, n_queries, n_distractors, token=None, seed=0):
    repo, rev = task.metadata.dataset["path"], task.metadata.dataset["revision"]
    subset, split = task.hf_subsets[0], task.eval_splits[-1]
    files = HfApi(token=token).list_repo_files(repo, repo_type="dataset", revision=rev)

    def parts(kind):
        found = sorted(f for f in files if f.startswith(f"{subset}-{kind}/") and f.endswith(".parquet"))
        return found if kind == "corpus" else [f for f in found if Path(f).name.startswith(split)]

    if not (parts("corpus") and parts("queries") and parts("qrels")):
        return False

    def get(f):
        return Path(hf_hub_download(repo, f, repo_type="dataset", revision=rev, token=token))

    qrels = pd.concat([pd.read_parquet(get(f)) for f in parts("qrels")])
    queries = pd.concat([pd.read_parquet(get(f)) for f in parts("queries")])
    queries = queries.rename(columns={"id": "_id"}) if "_id" not in queries.columns else queries
    rng = np.random.default_rng(seed)
    qids = set(rng.choice(sorted(qrels["query-id"].unique()), min(n_queries, qrels["query-id"].nunique()),
                          replace=False))
    qrels = qrels[qrels["query-id"].isin(qids)]
    judged, corpus = set(qrels["corpus-id"]), []
    corpus_files = [pq.ParquetFile(get(f)) for f in parts("corpus")]
    p_keep = n_distractors / max(1, sum(pf.metadata.num_rows for pf in corpus_files))
    for pf in corpus_files:
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg).to_pandas()
            idc = "_id" if "_id" in t.columns else "id"
            corpus.append(t[t[idc].isin(judged) | (rng.random(len(t)) < p_keep)].rename(columns={idc: "_id"}))
    corpus = pd.concat(corpus, ignore_index=True)
    title = corpus["title"].fillna("").tolist() if "title" in corpus.columns else [""] * len(corpus)
    queries = queries[queries["_id"].isin(qids)]
    relevant = defaultdict(dict)
    for r in qrels.itertuples():
        relevant[r[1]][r[2]] = int(r[3])
    task.hf_subsets = [subset]
    task.filter_eval_splits([split])
    task.dataset = {subset: {split: RetrievalSplitData(
        corpus=hf_datasets.Dataset.from_dict({"id": corpus["_id"].tolist(), "title": title,
                                              "text": corpus["text"].tolist()}),
        queries=hf_datasets.Dataset.from_dict({"id": queries["_id"].tolist(), "text": queries["text"].tolist()}),
        relevant_docs=dict(relevant), top_ranked=None)}}
    task.data_loaded = True
    log(f"  SMOKE subset of {task.metadata.name}: {len(queries)} queries, {len(corpus):,} documents")
    return True


def get_tasks(names, languages, smoke=False, smoke_queries=200, smoke_distractors=5_000, token=None):
    tasks = list(mteb.get_tasks(tasks=list(names), languages=list(languages)))
    for t in tasks:
        if smoke:
            smoke_subset(t, smoke_queries, smoke_distractors, token)
        else:
            log(f"  {t.metadata.name}: subsets {t.hf_subsets}, split {t.eval_splits} (full corpus)")
    return tasks


def run_mteb_tasks(encoder, tasks):
    t0 = time.time()
    result = mteb.evaluate(encoder, tasks, cache=None, overwrite_strategy="always", show_progress_bar=False,
                           encode_kwargs={"batch_size": 1024})
    out = {}
    for tr in result.task_results:
        for split, rows in tr.scores.items():
            for row in rows:
                out[tr.task_name] = {"ndcg@10": row["ndcg_at_10"], "recall@100": row["recall_at_100"],
                                     "subset": row.get("hf_subset"), "split": split}
    summary = ", ".join(f"{k} nDCG@10 {v['ndcg@10']:.4f} R@100 {v['recall@100']:.4f}" for k, v in out.items())
    log(f"  {encoder.mteb_model_meta.name}: {summary} ({time.time() - t0:.0f}s)")
    return out


def run_mteb_task(encoder, task):
    result = mteb.evaluate(encoder, [task], cache=None, overwrite_strategy="always", show_progress_bar=False,
                           encode_kwargs={"batch_size": 1024})
    tr = result.task_results[0]
    rows = sorted(((split, row) for split, rs in tr.scores.items() for row in rs), key=lambda x: x[0] != "test")
    split, row = rows[0]
    return {"ndcg@10": row["ndcg_at_10"], "recall@100": row["recall_at_100"], "subset": row.get("hf_subset"),
            "split": split}


def mteb_version():
    return mteb.__version__
