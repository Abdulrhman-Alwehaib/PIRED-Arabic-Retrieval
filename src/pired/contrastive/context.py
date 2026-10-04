import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

from ..common.device import setup_device
from ..common.notify import Announcer
from ..common.runlog import log
from ..encoder.embedding import Embedder, load_tokenizer, word_start_table
from ..encoder.weights import load_encoder_without_head
from ..tokenizer.fingerprint import check_fingerprint

ACTIVE = {}


def activate(stage):
    ACTIVE["stage"] = stage
    return stage


def active():
    return ACTIVE["stage"]


class Stage:
    def __init__(self, project, cfg, device=None):
        self.project, self.cfg, self.smoke = project, cfg, project.smoke
        self.device = device or setup_device()
        self.work = Path(cfg.data.work_dir)
        self.raw = self.work / "raw"
        self.clean = self.work / "clean"
        self.steps_dir = self.work / "steps"
        self.sources = {s.name: s for s in cfg.data.sources}
        self.source_names = list(self.sources)
        if cfg.data.spans_corpus == "fineweb2" and "spans" in self.sources:
            self.sources["spans"].description = "FineWeb-2 arb_Arab: short span -> longer span of the same document"
            self.sources["spans"].license = "ODC-By 1.0 (FineWeb-2; Common Crawl terms of use apply)"
        self.timings = {}
        self.loading = {}
        self.fs = HfFileSystem(token=project.hf_token)
        self.announce = Announcer("stage 3 contrastive", cfg.notify.telegram)
        self.tokenizer = None
        self.encoder = None
        self.embedder = None
        self.mlm_dir = None
        activate(self)

    @property
    def token(self):
        return self.project.hf_token

    @property
    def api(self):
        return self.project.api

    def repo_id(self, kind):
        h = self.cfg.hub
        explicit = {"data": h.data_repo, "checkpoints": h.checkpoint_repo, "model": h.model_repo}[kind]
        suffix = {"data": "weak-pairs", "checkpoints": "contrastive-checkpoints", "model": "contrastive-weak"}[kind]
        return self.project.repo_id(suffix, explicit, smoke_suffix=True)

    def local_file(self, local_name, path):
        return Path(self.cfg.data.raw_dir) / local_name / path

    def fetch_file(self, repo, path, local_name, revision=None):
        target = self.local_file(local_name, path)
        if target.exists():
            return target
        return Path(hf_hub_download(repo, path, repo_type="dataset", revision=revision,
                                    local_dir=Path(self.cfg.data.raw_dir) / local_name, token=self.token))

    @staticmethod
    def remote_path(repo, path, revision):
        return f"datasets/{repo}{'@' + revision.replace('/', '%2F') if revision else ''}/{path}"

    def open_parquet(self, repo, path, local_name, revision=None, download=None):
        target = self.local_file(local_name, path)
        if target.exists():
            return pq.ParquetFile(target)
        if self.smoke if download is None else not download:
            return pq.ParquetFile(self.fs.open(self.remote_path(repo, path, revision), "rb"))
        return pq.ParquetFile(self.fetch_file(repo, path, local_name, revision))

    def num_row_groups(self, repo, path, local_name, revision=None):
        target = self.local_file(local_name, path)
        if target.exists():
            return pq.ParquetFile(target).metadata.num_row_groups
        with self.fs.open(self.remote_path(repo, path, revision), "rb") as f:
            return pq.ParquetFile(f).metadata.num_row_groups

    def repo_parquet_files(self, repo, prefix, revision=None):
        entries = self.api.list_repo_tree(repo, path_in_repo=prefix, repo_type="dataset", revision=revision,
                                          recursive=True)
        return sorted(e.path for e in entries if e.path.endswith(".parquet"))

    def prefer_local(self, files, local_name):
        local = [f for f in files if self.local_file(local_name, f).exists()]
        return local if self.smoke and local else files

    def mlm_checkpoint_dir(self):
        m = self.cfg.model
        local = Path(m.mlm_checkpoint)
        source = {"model.safetensors": m.hub_weights, "config.json": m.hub_weights, "tokenizer.json": m.hub_extras,
                  "tokenizer_config.json": m.hub_extras, "reference_outputs.safetensors": m.hub_extras}
        missing = [name for name in source if not (local / name).exists()]
        if not missing:
            return local
        repo = self.project.repo_id("mlm-checkpoints", m.hub_repo)
        local.mkdir(parents=True, exist_ok=True)
        for name in missing:
            cached = hf_hub_download(repo, f"{source[name]}/{name}", token=self.token)
            shutil.copyfile(cached, local / f"{name}.tmp")
            os.replace(local / f"{name}.tmp", local / name)
            log(f"downloaded {repo}/{source[name]}/{name} -> {local / name}")
        return local

    def load_text_tools(self, folder=None):
        folder = Path(folder or self.mlm_checkpoint_dir())
        self.mlm_dir = folder
        self.tokenizer = load_tokenizer(folder)
        check_fingerprint(self.tokenizer, self.cfg.model.tokenizer_fingerprint, str(folder))
        self.normalizer = self.tokenizer.backend_tokenizer.normalizer
        m, f = self.cfg.model, self.cfg.filters
        self.q_prefix = m.query_prefix if m.use_prefixes else ""
        self.p_prefix = m.passage_prefix if m.use_prefixes else ""
        self.max_q, self.max_p = f.max_query_tokens, f.max_passage_tokens
        self.n_q_prefix = self.count_tokens(self.q_prefix)
        self.n_p_prefix = self.count_tokens(self.p_prefix)
        self.pad_id = self.tokenizer.pad_token_id
        self.is_word_start = word_start_table(self.tokenizer)
        self.check_prefixes_do_not_merge()
        return self.tokenizer

    def count_tokens(self, text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0

    def check_prefixes_do_not_merge(self):
        tok = self.tokenizer
        for prefix, n in ((self.q_prefix, self.n_q_prefix), (self.p_prefix, self.n_p_prefix)):
            for text in ["ما هي عاصمة فرنسا؟", "Paris 2024", "«مرحبا»"]:
                full = tok(prefix + text)["input_ids"]
                alone = tok(text, add_special_tokens=False)["input_ids"]
                if full[1:1 + n] != tok(prefix, add_special_tokens=False)["input_ids"] or full[1 + n:-1] != alone:
                    raise AssertionError(f"the prefix {prefix!r} merges with the text {text!r}")

    def normalize(self, text):
        return self.normalizer.normalize_str(text)

    def load_encoder(self):
        if self.tokenizer is None:
            self.load_text_tools()
        encoder, skipped = load_encoder_without_head(self.mlm_dir)
        self.encoder = encoder.to(self.device)
        self.embedder = self.make_embedder(self.encoder)
        return self.encoder, skipped

    def make_embedder(self, model):
        return Embedder(model, self.tokenizer, self.device, self.q_prefix, self.p_prefix, self.max_q, self.max_p)

    def tokenize_texts(self, texts, kind):
        return self.make_embedder(self.encoder).tokenize(texts, kind)
