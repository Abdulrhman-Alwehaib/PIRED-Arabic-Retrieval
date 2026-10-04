import gzip
import hashlib
import io
import json
import random
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download

from ..common.device import setup_device
from ..common.hub import pull_folder, push_folder
from ..common.io import write_parquet
from ..common.notify import Announcer
from ..common.runlog import log
from ..common.text import text_key
from ..encoder.embedding import Embedder, load_tokenizer, read_pooling_config, word_start_table
from ..encoder.weights import MODEL_FILES, check_reference_outputs, load_encoder
from ..tokenizer.fingerprint import check_fingerprint

ACTIVE = {}


def active():
    return ACTIVE["stage"]


class Stage:
    def __init__(self, project, cfg, device=None, with_encoder=False):
        self.project, self.cfg, self.smoke = project, cfg, project.smoke
        self.device = device or setup_device()
        self.work = Path(cfg.data.work_dir)
        self.p1 = self.work / "part1"
        self.raw1 = self.p1 / "raw"
        self.steps1 = self.p1 / "steps"
        self.final1 = self.p1 / "final"
        self.syn = self.p1 / "synthetic"
        self.corpora = self.work / "corpora"
        self.p2 = self.work / "part2"
        self.final2 = self.p2 / "final"
        self.sources = {s.name: s for s in cfg.data.sources}
        self.source_names = list(self.sources)
        self.timings = {}
        self.loading = {}
        self.fs = HfFileSystem(token=project.hf_token)
        self.announce = Announcer("05", cfg.notify.telegram and not project.smoke)
        self.encoder = None
        ACTIVE["stage"] = self
        self.stage3_dir = self.stage3_model_dir()
        self.load_text_tools()
        if with_encoder:
            self.load_encoder()

    @property
    def token(self):
        return self.project.hf_token

    @property
    def api(self):
        return self.project.api

    def stop(self, message):
        self.announce.stop(message)

    def repo_id(self, kind):
        h = self.cfg.hub
        explicit = {"queries": h.queries_repo, "pairs": h.pairs_repo, "checkpoints": h.checkpoint_repo,
                    "model": h.model_repo}[kind]
        suffix = {"queries": "supervised-queries", "pairs": "supervised-pairs", "checkpoints": "supervised-checkpoints",
                  "model": "supervised"}[kind]
        return self.project.repo_id(suffix, explicit, smoke_suffix=True)

    def push_folder(self, folder, repo_id, repo_type):
        return push_folder(self.api, folder, repo_id, repo_type, self.cfg.hub.push)

    def pull_folder(self, repo_id, repo_type, local):
        return pull_folder(repo_id, repo_type, local, self.token)

    def stage3_model_dir(self):
        m = self.cfg.model
        local = Path(m.stage3_dir)
        missing = [f for f in MODEL_FILES if not (local / f).exists()]
        if missing:
            repo = self.project.repo_id("contrastive-weak", m.stage3_repo)
            snapshot_download(repo, allow_patterns=missing, local_dir=local, token=self.token)
            log(f"stage-3 model: downloaded {missing} from {repo} to {local}")
        return local

    def load_text_tools(self):
        cfg = self.cfg
        self.tokenizer = load_tokenizer(self.stage3_dir)
        self.normalizer = self.tokenizer.backend_tokenizer.normalizer
        self.tokenizer_fingerprint = check_fingerprint(self.tokenizer, cfg.model.tokenizer_fingerprint,
                                                       str(self.stage3_dir))
        pooling = read_pooling_config(self.stage3_dir)
        self.q_prefix, self.p_prefix = cfg.model.query_prefix, cfg.model.passage_prefix
        self.max_q, self.max_p = cfg.filters.max_query_tokens, cfg.filters.max_passage_tokens
        if (pooling["query_prefix"], pooling["passage_prefix"]) != (self.q_prefix, self.p_prefix):
            raise AssertionError(f"stage-3 prefixes differ from the config: {pooling}")
        if (pooling["max_query_tokens"], pooling["max_passage_tokens"]) != (self.max_q, self.max_p):
            raise AssertionError(f"stage-3 limits differ from the config: {pooling}")
        if pooling["pooling"] != "mean" or pooling["normalize"] != "l2":
            raise AssertionError(f"stage-3 pooling is not mean + l2: {pooling}")
        self.n_q_prefix = len(self.tokenizer(self.q_prefix, add_special_tokens=False)["input_ids"])
        self.n_p_prefix = len(self.tokenizer(self.p_prefix, add_special_tokens=False)["input_ids"])
        self.pad_id = self.tokenizer.pad_token_id
        self.is_word_start = word_start_table(self.tokenizer)
        if len(self.tokenizer("كلمة " * 600)["input_ids"]) <= 600:
            raise AssertionError("the tokenizer still truncates silently")

    def load_encoder(self):
        self.encoder = load_encoder(self.stage3_dir).to(self.device).eval()
        diffs = check_reference_outputs(self.encoder, self.stage3_dir, atol=1e-4)
        log(f"encoder: {sum(p.numel() for p in self.encoder.parameters()) / 1e6:.2f}M parameters, strict load; "
            f"reference outputs reproduced (max |diff| {max(diffs.values()):.1e})")
        self.embedder = self.make_embedder(self.encoder)
        return self.encoder

    def make_embedder(self, model):
        return Embedder(model, self.tokenizer, self.device, self.q_prefix, self.p_prefix, self.max_q, self.max_p)

    def tokenize_texts(self, texts, kind):
        return Embedder(None, self.tokenizer, self.device, self.q_prefix, self.p_prefix, self.max_q,
                        self.max_p).tokenize(texts, kind)

    def normalize(self, text):
        return self.normalizer.normalize_str(text)

    def text_key(self, text):
        return text_key(self.normalize(text))

    def fetch_file(self, repo, path, local_name, revision=None):
        target = Path(self.cfg.data.raw_dir) / local_name / path
        if target.exists():
            return target
        return Path(hf_hub_download(repo, path, repo_type="dataset", revision=revision,
                                    local_dir=Path(self.cfg.data.raw_dir) / local_name, token=self.token))

    def open_lines(self, repo, path, local_name, gz=False):
        local = Path(self.cfg.data.raw_dir) / local_name / path
        if self.smoke and not local.exists():
            raw = self.fs.open(f"datasets/{repo}/{path}", "rb")
        else:
            raw = open(self.fetch_file(repo, path, local_name), "rb")
        return io.TextIOWrapper(gzip.GzipFile(fileobj=raw) if gz else raw, encoding="utf-8")

    def read_qrels(self, repo, path, local_name):
        return pd.read_csv(self.fetch_file(repo, path, local_name), sep=r"\s+", header=None,
                           names=["qid", "q0", "docid", "rel"], dtype=str).assign(rel=lambda d: d["rel"].astype(int))

    def read_topics(self, repo, path, local_name):
        lines = self.fetch_file(repo, path, local_name).read_text(encoding="utf-8").splitlines()
        return {a: b for a, b in (line.split("\t", 1) for line in lines if "\t" in line)}

    def miracl_corpus_files(self):
        d = self.cfg.data
        entries = self.api.list_repo_tree(d.miracl_corpus_repo, path_in_repo=d.miracl_corpus_dir, repo_type="dataset")
        return sorted(e.path for e in entries if e.path.endswith(".jsonl.gz"))

    def iter_miracl_corpus(self, keep=None):
        d = self.cfg.data
        for f in self.miracl_corpus_files():
            with gzip.open(self.fetch_file(d.miracl_corpus_repo, f, "miracl-corpus"), "rt", encoding="utf-8") as fh:
                for line in fh:
                    docid = line[11: line.find('"', 11)] if line.startswith('{"docid": "') else json.loads(line)["docid"]
                    if keep is None or keep(docid):
                        r = json.loads(line)
                        yield r["docid"], r.get("title") or "", r.get("text") or ""

    def miracl_corpus_frame(self):
        path = self.corpora / ("miracl-smoke.parquet" if self.smoke else "miracl.parquet")
        if not path.exists():
            t0 = time.time()
            keep = None
            if self.smoke:
                def keep(docid):
                    return int(hashlib.md5(docid.encode()).hexdigest()[:8], 16) % 1000 < 8
            df = pd.DataFrame(list(self.iter_miracl_corpus(keep)), columns=["docid", "title", "text"])
            df.insert(1, "article", df["docid"].str.split("#").str[0])
            write_parquet(df, path)
            log(f"MIRACL corpus -> {path}: {len(df):,} passages ({time.time() - t0:.0f}s)")
        return pd.read_parquet(path)

    def read_mmarco_collection(self, keep=None, max_bytes=None):
        d = self.cfg.data
        if max_bytes is not None and not (Path(d.raw_dir) / "mmarco" / d.mmarco_collection).exists():
            with self.fs.open(f"datasets/{d.mmarco_repo}/{d.mmarco_collection}", "rb", block_size=max_bytes) as f:
                lines = f.read(max_bytes).decode("utf-8", "ignore").split("\n")[:-1]
        else:
            lines = open(self.fetch_file(d.mmarco_repo, d.mmarco_collection, "mmarco"), encoding="utf-8")
        for line in lines:
            pid, _, text = line.rstrip("\n").partition("\t")
            if pid.isdigit() and (keep is None or keep(int(pid))):
                yield int(pid), text

    def news_leads(self, n_needed):
        d = self.cfg.data
        repo = self.project.repo_id("weak-pairs", d.news_repo)
        files = sorted(e.path for e in self.api.list_repo_tree(repo, path_in_repo=d.news_prefix, repo_type="dataset")
                       if e.path.endswith(".parquet"))
        random.Random(d.seed).shuffle(files)
        parts, total = [], 0
        for f in files:
            if self.smoke:
                with self.fs.open(f"datasets/{repo}/{f}", "rb") as fh:
                    t = pq.ParquetFile(fh).read_row_group(0, columns=["passage"]).to_pandas()
            else:
                t = pd.read_parquet(self.fetch_file(repo, f, "weak-pairs"), columns=["passage"])
            parts.append(t)
            total += len(t)
            if total >= n_needed:
                break
        df = pd.concat(parts, ignore_index=True).rename(columns={"passage": "text"})
        df["passage"], df["title"] = df["text"], ""
        df["article"] = [f"news-{i}" for i in range(len(df))]
        return df, f"{repo} {d.news_prefix}/*.parquet ({len(parts)} of {len(files)} files)"
