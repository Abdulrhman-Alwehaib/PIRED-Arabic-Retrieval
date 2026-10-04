import contextlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from ..common.runlog import log
from ..tokenizer.fingerprint import vocab_fingerprint


def missing_shards(local_dir):
    manifest = Path(local_dir) / "manifest.json"
    if not manifest.exists():
        return None
    shards = json.loads(manifest.read_text(encoding="utf-8"))["shards"]
    return [s["file"] for s in shards if not (Path(local_dir) / s["file"]).exists()]


def download_pretraining_data(local_dir, repo_id, token, delete_consumed_shards=False):
    local_dir = Path(local_dir)
    if missing_shards(local_dir) == [] and (local_dir / "validation.npy").exists():
        log(f"pretraining data already in {local_dir}")
        return
    if repo_id is None:
        raise RuntimeError("the pretraining data is not on this machine and HF_TOKEN is not set")
    if delete_consumed_shards:
        hf_hub_download(repo_id, "manifest.json", repo_type="dataset", local_dir=local_dir, token=token)
        log(f"delete_consumed_shards is on: shards are downloaded from {repo_id} during training")
        return
    wanted = ("manifest.json", "validation.npy")
    sizes = {s.rfilename: s.size or 0 for s in HfApi(token=token).dataset_info(repo_id, files_metadata=True).siblings
             if s.rfilename in wanted or s.rfilename.startswith("shards/")}
    need_gb = sum(size for name, size in sizes.items() if not (local_dir / name).exists()) / 1e9
    local_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(local_dir).free / 1e9
    if free_gb < need_gb * 1.05:
        raise RuntimeError(f"need {need_gb:.1f} GB but only {free_gb:.1f} GB free in {local_dir}; free space or turn "
                           "on delete_consumed_shards")
    log(f"downloading {need_gb:.1f} GB from {repo_id} to {local_dir} ({free_gb:.0f} GB free)")
    t0 = time.time()
    snapshot_download(repo_id, repo_type="dataset", local_dir=local_dir, token=token, max_workers=16,
                      allow_patterns=[*wanted, "shards/*.npy"])
    missing = missing_shards(local_dir)
    if missing or not (local_dir / "validation.npy").exists():
        raise RuntimeError(f"download incomplete ({len(missing or [])} shards missing); run it again")
    n_shards = len(json.loads((local_dir / "manifest.json").read_text(encoding="utf-8"))["shards"])
    log(f"done in {(time.time() - t0) / 60:.1f} min: manifest, validation rows and {n_shards} shards")


class PretrainingData:
    def __init__(self, tokenizer, seq_len, micro_batch_size, seed, settings, repo_id=None, token=None):
        self.local_dir = Path(settings.local_dir)
        self.repo_id, self.token = repo_id, token
        self.seq_len, self.micro_batch_size, self.seed = seq_len, micro_batch_size, seed
        self.window_shards = settings.window_shards
        self.delete_consumed = settings.delete_consumed_shards and repo_id is not None
        self.manifest = json.loads(self._fetch("manifest.json").read_text(encoding="utf-8"))
        if self.manifest["seq_len"] != seq_len:
            raise ValueError(f"the shards have seq_len {self.manifest['seq_len']}, training uses {seq_len}")
        if self.manifest["vocab_fingerprint"] != vocab_fingerprint(tokenizer.get_vocab()):
            raise ValueError("the shards were tokenized with a different tokenizer")
        self.shards = [s for s in self.manifest["shards"] if s["n_sequences"] > 0]
        self.sizes = np.array([s["n_sequences"] for s in self.shards], dtype=np.int64)
        self.total = int(self.sizes.sum())
        self.position = 0
        self._layouts = {}
        self._window = None
        self._order = None
        self._arrays = {}
        self._prefetcher = ThreadPoolExecutor(max_workers=2)

    def __iter__(self):
        return self

    def describe(self):
        fallback = f", missing shards come from {self.repo_id}" if self.repo_id else ""
        return (f"{self.total:,} sequences = {self.total * self.seq_len / 1e9:.2f}B tokens in {len(self.shards)} "
                f"shards (local: {self.local_dir}{fallback})")

    def _fetch(self, rel):
        path = self.local_dir / rel
        if path.exists():
            return path
        if not self.repo_id:
            raise FileNotFoundError(f"{path} is missing and no Hub dataset repo is configured")
        return Path(hf_hub_download(self.repo_id, rel, repo_type="dataset", local_dir=self.local_dir, token=self.token))

    def _layout(self, epoch):
        if epoch not in self._layouts:
            order = np.random.default_rng([self.seed, epoch]).permutation(len(self.shards))
            windows = [order[i: i + self.window_shards] for i in range(0, len(order), self.window_shards)]
            self._layouts[epoch] = (windows, np.cumsum([self.sizes[w].sum() for w in windows]))
        return self._layouts[epoch]

    def _enter_window(self, epoch, w):
        windows, _ = self._layout(epoch)
        shard_ids = windows[w]
        finished = set(self._arrays) - set(shard_ids.tolist())
        self._arrays = {int(s): np.load(self._fetch(self.shards[s]["file"]), mmap_mode="r") for s in shard_ids}
        if self.delete_consumed:
            for s in finished:
                with contextlib.suppress(OSError):
                    (self.local_dir / self.shards[s]["file"]).unlink()
        owners = np.repeat(shard_ids, self.sizes[shard_ids])
        rows = np.concatenate([np.arange(self.sizes[s]) for s in shard_ids])
        perm = np.random.default_rng([self.seed, epoch, w]).permutation(len(rows))
        self._order = (owners[perm], rows[perm])
        self._window = (epoch, w)
        upcoming = windows[w + 1] if w + 1 < len(windows) else self._layout(epoch + 1)[0][0]
        for s in upcoming:
            self._prefetcher.submit(self._fetch, self.shards[s]["file"])

    def __next__(self):
        batch = np.empty((self.micro_batch_size, self.seq_len), dtype=np.uint16)
        for j in range(self.micro_batch_size):
            epoch, within = divmod(self.position, self.total)
            _, ends = self._layout(epoch)
            w = int(np.searchsorted(ends, within, side="right"))
            if self._window != (epoch, w):
                self._enter_window(epoch, w)
            offset = within - (int(ends[w - 1]) if w else 0)
            batch[j] = self._arrays[int(self._order[0][offset])][self._order[1][offset]]
            self.position += 1
        return batch

    def close(self):
        self._arrays, self._order, self._window = {}, None, None
        self._prefetcher.shutdown(wait=True)

    def validation_sequences(self, max_sequences):
        return np.array(np.load(self._fetch(self.manifest["validation"]["file"]), mmap_mode="r")[:max_sequences])

    def state_dict(self):
        return {"seed": self.seed, "position": self.position, "total_sequences": self.total,
                "window_shards": self.window_shards}

    def load_state_dict(self, state):
        if state["total_sequences"] != self.total:
            raise ValueError(f"the dataset changed since this checkpoint ({state['total_sequences']:,} sequences "
                             f"then, {self.total:,} now); the saved position would point elsewhere")
        self.seed, self.position, self.window_shards = state["seed"], state["position"], state["window_shards"]
        self._layouts, self._window = {}, None


class SyntheticData:
    def __init__(self, vocab_size, first_regular_id, cls_id, sep_id, seq_len, micro_batch_size, seed=0):
        self.vocab_size, self.first_regular_id, self.cls_id, self.sep_id = vocab_size, first_regular_id, cls_id, sep_id
        self.seq_len, self.micro_batch_size, self.seed = seq_len, micro_batch_size, seed
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.position)
        self.position += 1
        lengths = torch.randint(self.seq_len // 2, self.seq_len + 1, (self.micro_batch_size,), generator=g)
        return [[self.cls_id, *torch.randint(self.first_regular_id, self.vocab_size, (int(n) - 2,), generator=g).tolist(),
                 self.sep_id] for n in lengths]

    def state_dict(self):
        return {"seed": self.seed, "position": self.position}

    def load_state_dict(self, state):
        self.seed, self.position = state["seed"], state["position"]


class PackedTextData:
    def __init__(self, docs_token_ids, seq_len, micro_batch_size, cls_id, sep_id):
        stream = []
        for ids in docs_token_ids:
            stream.extend(ids)
            stream.append(sep_id)
        body = seq_len - 2
        self.sequences = [[cls_id, *stream[i: i + body], sep_id] for i in range(0, len(stream) - body + 1, body)]
        if not self.sequences:
            raise ValueError("not enough text for a single sequence")
        self.micro_batch_size = micro_batch_size
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        n = len(self.sequences)
        start = self.position * self.micro_batch_size
        self.position += 1
        return [self.sequences[(start + j) % n] for j in range(self.micro_batch_size)]

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]
