import threading
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from ..common.device import autocast_ctx
from ..common.io import read_json
from .weights import load_encoder

QUERY_PREFIX = "استعلام: "
PASSAGE_PREFIX = "نص: "
MAX_QUERY_TOKENS = 64
MAX_PASSAGE_TOKENS = 256
POOLING_FILE = "pooling_config.json"
WORD_START = chr(0x2581)
TOKENIZER_LOCK = threading.RLock()


def round_up(n, multiple=8):
    return -(-n // multiple) * multiple


def pad_chunks(id_lists, chunk_tokens, pad_id, mask_full_chunks=False):
    order = sorted(range(len(id_lists)), key=lambda i: -len(id_lists[i]))
    chunks, i = [], 0
    while i < len(order):
        width = round_up(len(id_lists[order[i]]))
        rows = order[i: i + max(1, chunk_tokens // width)]
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for r, j in enumerate(rows):
            ids[r, : len(id_lists[j])] = torch.tensor(id_lists[j])
            mask[r, : len(id_lists[j])] = 1
        keep_mask = mask_full_chunks or not bool(mask.all())
        chunks.append((torch.tensor(rows), ids, mask if keep_mask else None))
        i += len(rows)
    return chunks


def load_tokenizer(folder):
    tokenizer = AutoTokenizer.from_pretrained(folder)
    if tokenizer.backend_tokenizer.truncation is not None:
        tokenizer.backend_tokenizer.no_truncation()
    return tokenizer


def word_start_table(tokenizer):
    table = np.zeros(len(tokenizer), dtype=bool)
    for token, token_id in tokenizer.get_vocab().items():
        table[token_id] = token.startswith(WORD_START)
    return table


def read_pooling_config(folder):
    return read_json(Path(folder) / POOLING_FILE)


def pooling_mismatches(pooling, query_prefix=QUERY_PREFIX, passage_prefix=PASSAGE_PREFIX,
                       max_query_tokens=MAX_QUERY_TOKENS, max_passage_tokens=MAX_PASSAGE_TOKENS):
    want = {"query_prefix": query_prefix, "passage_prefix": passage_prefix, "max_query_tokens": max_query_tokens,
            "max_passage_tokens": max_passage_tokens, "pooling": "mean", "normalize": "l2"}
    return {k: pooling.get(k) for k, v in want.items() if pooling.get(k) != v}


class Embedder:
    def __init__(self, model, tokenizer, device=None, query_prefix=QUERY_PREFIX, passage_prefix=PASSAGE_PREFIX,
                 max_query_tokens=MAX_QUERY_TOKENS, max_passage_tokens=MAX_PASSAGE_TOKENS, mask_full_chunks=False):
        self.model, self.tokenizer = model, tokenizer
        self.device = device or next(model.parameters()).device
        self.prefixes = {"query": query_prefix, "passage": passage_prefix}
        self.limits = {"query": max_query_tokens, "passage": max_passage_tokens}
        self.mask_full_chunks = mask_full_chunks

    @classmethod
    def from_folder(cls, folder, device=None, **kwargs):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pooling = read_pooling_config(folder) if (Path(folder) / POOLING_FILE).exists() else {}
        settings = {k: pooling[k] for k in ("query_prefix", "passage_prefix", "max_query_tokens", "max_passage_tokens")
                    if k in pooling}
        model = load_encoder(folder).to(device).eval()
        return cls(model, load_tokenizer(folder), device, **{**settings, **kwargs})

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id

    def prefix_length(self, kind):
        prefix = self.prefixes[kind]
        return len(self.tokenizer(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0

    def tokenize(self, texts, kind):
        prefix = self.prefixes[kind]
        with TOKENIZER_LOCK:
            return self.tokenizer([prefix + t for t in texts], add_special_tokens=True, truncation=True,
                                  max_length=self.limits[kind])["input_ids"]

    @torch.no_grad()
    def embed_ids(self, id_lists, batch_tokens, dtype=torch.float32):
        model = self.model
        was_training = model.training
        model.eval()
        out = torch.empty((len(id_lists), model.config.hidden_size), dtype=dtype, device=self.device)
        for rows, ids, mask in pad_chunks(id_lists, batch_tokens, self.pad_id, self.mask_full_chunks):
            with autocast_ctx(self.device):
                vectors = model.embed(ids.to(self.device), None if mask is None else mask.to(self.device))
            out[rows.to(self.device)] = vectors.to(dtype)
        model.train(was_training)
        return out

    def encode(self, texts, kind, batch_tokens=32_768, dtype=torch.float32):
        return self.embed_ids(self.tokenize(list(texts), kind), batch_tokens, dtype)

    def encode_queries(self, texts, batch_tokens=32_768):
        return self.encode(texts, "query", batch_tokens)

    def encode_passages(self, texts, batch_tokens=32_768):
        return self.encode(texts, "passage", batch_tokens)
