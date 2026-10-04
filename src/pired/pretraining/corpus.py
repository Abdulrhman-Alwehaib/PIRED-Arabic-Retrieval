import dataclasses
import hashlib
import json
import os
import random
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from huggingface_hub.hf_api import RepoFile
from tokenizers import Tokenizer

from ..common.runlog import log
from ..tokenizer.fingerprint import vocab_fingerprint
from ..tokenizer.normalization import METASPACE


@dataclass
class ParquetFile:
    source: str
    repo_id: str
    path: str
    size_bytes: int
    num_rows: int
    num_row_groups: int
    text_bytes: int
    est_tokens: float = 0.0


@dataclass
class Unit:
    uid: str
    source: str
    repo_id: str
    path: str
    text_field: str
    rg_start: int
    rg_end: int


def is_arabic_letter(c):
    o = ord(c)
    return c.isalpha() and (0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF)


def is_latin_letter(c):
    return c.isalpha() and (c.isascii() or 0x00C0 <= ord(c) <= 0x024F)


def fmt_tokens(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def save_npy(path, array):
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def pack(stream_parts, seq_len, cls_id, sep_id):
    if not stream_parts:
        return np.zeros((0, seq_len), dtype=np.uint16)
    stream = np.concatenate(stream_parts)
    body = seq_len - 2
    n = len(stream) // body
    rows = np.empty((n, seq_len), dtype=np.uint16)
    rows[:, 0], rows[:, -1] = cls_id, sep_id
    rows[:, 1:-1] = stream[: n * body].reshape(n, body)
    return rows


class CorpusBuilder:
    def __init__(self, cfg, token=None):
        self.cfg, self.token = cfg, token
        self.tokenizer_file = Path(cfg.tokenizer_dir) / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_file))
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = self.tokenizer.get_vocab_size()
        if self.vocab_size > 2**16:
            raise ValueError("uint16 shards need a vocabulary of at most 65,536 tokens")
        self.cls_id, self.sep_id = self.vocab["[CLS]"], self.vocab["[SEP]"]
        self.sep_array = np.array([self.sep_id], dtype=np.uint16)
        self.arabic_word_start = np.zeros(self.vocab_size, dtype=bool)
        self.latin_word_start = np.zeros(self.vocab_size, dtype=bool)
        for token, token_id in self.vocab.items():
            if token.startswith(METASPACE) and len(token) > 1:
                self.arabic_word_start[token_id] = is_arabic_letter(token[1])
                self.latin_word_start[token_id] = is_latin_letter(token[1])
        self.fingerprint = vocab_fingerprint(self.vocab)
        self.out_dir = Path(cfg.out_dir)
        self.work_dir = self.out_dir / "work"
        for sub in ("hashes", "keep", "validation", "stats"):
            (self.work_dir / sub).mkdir(parents=True, exist_ok=True)
        (self.out_dir / "shards").mkdir(parents=True, exist_ok=True)
        self._thread_state = threading.local()
        self.plan = None
        self.units = []

    def fs(self):
        return HfFileSystem(token=self.token)

    def read_footer(self, repo_id, path, text_field):
        with self.fs().open(f"datasets/{repo_id}/{path}") as f:
            md = pq.ParquetFile(f).metadata
        text_col = [md.schema.column(i).name for i in range(md.num_columns)].index(text_field)
        text_bytes = sum(md.row_group(i).column(text_col).total_uncompressed_size for i in range(md.num_row_groups))
        return md.num_rows, md.num_row_groups, text_bytes

    def bytes_per_token(self, repo_id, path, text_field):
        with self.fs().open(f"datasets/{repo_id}/{path}") as f:
            pf = pq.ParquetFile(f)
            text_col = pf.schema_arrow.get_field_index(text_field)
            texts = pf.read_row_group(0, columns=[text_field]).column(0).to_pylist()
            rg_bytes = pf.metadata.row_group(0).column(text_col).total_uncompressed_size
        encoded = self.tokenizer.encode_batch_fast([t or "" for t in texts], add_special_tokens=False)
        return rg_bytes / max(sum(len(e.ids) for e in encoded), 1)

    def plan_sources(self):
        cfg, api = self.cfg, HfApi(token=self.token)
        plan = {"files": [], "sources": {}}
        for source in cfg.sources:
            listing = [e for e in api.list_repo_tree(source.repo_id, path_in_repo=source.prefix, repo_type="dataset")
                       if isinstance(e, RepoFile) and e.path.endswith(".parquet")]
            with ThreadPoolExecutor(max_workers=16) as pool:
                footers = list(pool.map(lambda e: self.read_footer(source.repo_id, e.path, source.text_field), listing))
            files = [ParquetFile(source.name, source.repo_id, e.path, e.size, *ft) for e, ft in zip(listing, footers)]
            bpt = self.bytes_per_token(source.repo_id, files[0].path, source.text_field)
            for f in files:
                f.est_tokens = f.text_bytes / bpt
            budget = source.share * cfg.target_tokens * (1 + cfg.margin)
            chosen, planned = [], 0.0
            for f in random.Random(f"{cfg.seed}-{source.name}").sample(files, len(files)):
                if planned >= budget:
                    break
                chosen.append(f)
                planned += f.est_tokens
            chosen.sort(key=lambda f: f.path)
            plan["files"] += [dataclasses.asdict(f) for f in chosen]
            plan["sources"][source.name] = {
                "available_files": len(files), "available_tokens": sum(f.est_tokens for f in files),
                "budget_tokens": budget, "chosen_files": len(chosen), "planned_tokens": planned,
                "download_bytes": sum(f.size_bytes for f in chosen), "bytes_per_token": bpt,
            }
        return plan

    def plan_settings(self):
        cfg = self.cfg
        return {"target_tokens": cfg.target_tokens, "margin": cfg.margin, "seed": cfg.seed,
                "sources": [dataclasses.asdict(s) for s in cfg.sources], "vocab": self.fingerprint}

    def make_units(self, plan):
        cfg = self.cfg
        text_field = {s.name: s.text_field for s in cfg.sources}
        units = []
        for f in plan["files"]:
            stem = Path(f["path"]).stem
            starts = list(range(0, f["num_row_groups"], cfg.unit_row_groups))
            if cfg.max_units_per_file is not None:
                starts = starts[: cfg.max_units_per_file]
            for start in starts:
                end = min(start + cfg.unit_row_groups, f["num_row_groups"])
                units.append(Unit(f"{f['source']}-{stem}-rg{start:05d}", f["source"], f["repo_id"], f["path"],
                                  text_field[f["source"]], start, end))
        return units

    def load_plan(self):
        plan_file = self.work_dir / "plan.json"
        plan = json.loads(plan_file.read_text(encoding="utf-8")) if plan_file.exists() else None
        settings = self.plan_settings()
        if plan is None or plan["settings"] != settings:
            plan = {"settings": settings, **self.plan_sources()}
            plan_file.write_text(json.dumps(plan, indent=1), encoding="utf-8")
        self.plan, self.units = plan, self.make_units(plan)
        return plan

    def plan_summary(self):
        lines = [f"{'source':<10}{'files':>12}{'available':>12}{'budget':>10}{'planned':>10}{'download':>11}"
                 f"{'bytes/tok':>11}"]
        for name, s in self.plan["sources"].items():
            lines.append(f"{name:<10}{s['chosen_files']:>5} / {s['available_files']:<5}"
                         f"{fmt_tokens(s['available_tokens']):>12}{fmt_tokens(s['budget_tokens']):>10}"
                         f"{fmt_tokens(s['planned_tokens']):>10}{s['download_bytes'] / 1e9:>8.1f} GB"
                         f"{s['bytes_per_token']:>11.2f}")
        planned = sum(s["planned_tokens"] for s in self.plan["sources"].values())
        download = sum(s["download_bytes"] for s in self.plan["sources"].values())
        lines.append(f"planned {fmt_tokens(planned)} tokens in {len(self.units)} work units; download "
                     f"{download / 1e9:.1f} GB; shards about {planned * 2 / 1e9:.1f} GB")
        free_gb = shutil.disk_usage(self.out_dir).free / 1e9
        if free_gb < (download + planned * 2) / 1e9 * 1.1:
            lines.append(f"WARNING: only {free_gb:.0f} GB free disk")
        return "\n".join(lines)

    def local_parquet(self, source, path):
        return Path(self.cfg.raw_dir) / source / path

    def download_file(self, f):
        return Path(hf_hub_download(f["repo_id"], f["path"], repo_type="dataset",
                                    local_dir=Path(self.cfg.raw_dir) / f["source"], token=self.token))

    def download(self):
        if self.cfg.remote_read:
            log("remote_read is on: parquet files are read from the Hub in place")
            return
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.cfg.download_workers) as pool:
            for done, future in enumerate(as_completed([pool.submit(self.download_file, f)
                                                        for f in self.plan["files"]]), 1):
                future.result()
                if done % 10 == 0 or done == len(self.plan["files"]):
                    log(f"download: {done}/{len(self.plan['files'])}")
        log(f"{len(self.plan['files'])} files ready in {self.cfg.raw_dir} ({time.time() - t0:.0f}s)")

    def open_unit_file(self, unit):
        local = self.local_parquet(unit.source, unit.path)
        if local.exists() or not self.cfg.remote_read:
            return pq.ParquetFile(local)
        return pq.ParquetFile(self.fs().open(f"datasets/{unit.repo_id}/{unit.path}", "rb"))

    def unit_texts(self, unit):
        pf = self.open_unit_file(unit)
        for rg in range(unit.rg_start, unit.rg_end):
            yield [t or "" for t in pf.read_row_group(rg, columns=[unit.text_field]).column(0).to_pylist()]

    def hash_unit(self, unit):
        out = self.work_dir / "hashes" / f"{unit.uid}.npz"
        if out.exists():
            return 0
        keys, validation = [], []
        for texts in self.unit_texts(unit):
            for text in texts:
                digest = hashlib.md5(text.encode("utf-8")).digest()
                keys.append(int.from_bytes(digest[:8], "big"))
                validation.append(int.from_bytes(digest, "big") % self.cfg.validation_every == 0)
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, keys=np.array(keys, dtype=np.uint64), validation=np.array(validation, dtype=bool))
        os.replace(tmp, out)
        return len(keys)

    def hash_all(self):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.cfg.hash_workers) as pool:
            n_hashed = sum(pool.map(self.hash_unit, self.units))
        log(f"hashed {n_hashed:,} new documents ({time.time() - t0:.0f}s)")
        return n_hashed

    def dedup(self):
        keys = [np.load(self.work_dir / "hashes" / f"{u.uid}.npz")["keys"] for u in self.units]
        all_keys = np.concatenate(keys)
        _, first = np.unique(all_keys, return_index=True)
        keep = np.zeros(len(all_keys), dtype=bool)
        keep[first] = True
        report = {}
        for u, part in zip(self.units, np.split(keep, np.cumsum([len(k) for k in keys])[:-1])):
            np.save(self.work_dir / "keep" / f"{u.uid}.npy", part)
            r = report.setdefault(u.source, {"docs": 0, "duplicates": 0})
            r["docs"] += len(part)
            r["duplicates"] += int((~part).sum())
        for name, r in report.items():
            log(f"{name:<10} {r['docs']:>12,} docs, {r['duplicates']:>10,} exact duplicates "
                f"({r['duplicates'] / max(r['docs'], 1):.2%})")
        return report

    def finished_units(self):
        return [u for u in self.units if (self.work_dir / "stats" / f"{u.uid}.json").exists()]

    def thread_tokenizer(self):
        if not hasattr(self._thread_state, "tokenizer"):
            self._thread_state.tokenizer = Tokenizer.from_file(str(self.tokenizer_file))
        return self._thread_state.tokenizer

    def tokenize_unit(self, unit):
        cfg = self.cfg
        stats_file = self.work_dir / "stats" / f"{unit.uid}.json"
        if stats_file.exists():
            return json.loads(stats_file.read_text(encoding="utf-8"))
        tok = self.thread_tokenizer()
        keep = np.load(self.work_dir / "keep" / f"{unit.uid}.npy")
        is_validation = np.load(self.work_dir / "hashes" / f"{unit.uid}.npz")["validation"]
        stats = Counter()
        train_parts, validation_parts = [], []
        i = 0
        for texts in self.unit_texts(unit):
            batch, batch_validation = [], []
            for text in texts:
                if not keep[i]:
                    stats["duplicate"] += 1
                elif len(text) < cfg.min_chars:
                    stats["too_short"] += 1
                else:
                    batch.append(text)
                    batch_validation.append(bool(is_validation[i]))
                i += 1
            stats["docs"] += len(texts)
            for start in range(0, len(batch), cfg.encode_batch_docs):
                chunk = slice(start, start + cfg.encode_batch_docs)
                id_lists = [enc.ids for enc in tok.encode_batch_fast(batch[chunk], add_special_tokens=False)]
                for id_list, to_validation in zip(id_lists, batch_validation[chunk]):
                    ids = np.asarray(id_list, dtype=np.uint16)
                    if len(ids) < cfg.min_tokens:
                        stats["too_short"] += 1
                        continue
                    arabic = int(self.arabic_word_start[ids].sum())
                    latin = int(self.latin_word_start[ids].sum())
                    if arabic + latin and arabic < cfg.min_arabic_word_ratio * (arabic + latin):
                        stats["non_arabic"] += 1
                        continue
                    parts = validation_parts if to_validation else train_parts
                    parts += [ids, self.sep_array]
                    prefix = "validation" if to_validation else "kept"
                    stats[f"{prefix}_docs"] += 1
                    stats[f"{prefix}_tokens"] += len(ids)
        if i != len(keep):
            raise AssertionError(f"{unit.uid}: row count changed since the hash pass")
        rows = pack(train_parts, cfg.seq_len, self.cls_id, self.sep_id)
        save_npy(self.out_dir / "shards" / f"{unit.uid}.npy", rows)
        save_npy(self.work_dir / "validation" / f"{unit.uid}.npy",
                 np.concatenate(validation_parts) if validation_parts else np.zeros(0, dtype=np.uint16))
        result = {**stats, "sequences": len(rows), "source": unit.source}
        stats_file.write_text(json.dumps(result), encoding="utf-8")
        return result

    def tokenize_all(self):
        units = self.finished_units() if self.cfg.finished_units_only else self.units
        t0 = time.time()
        unit_stats = {}
        with ThreadPoolExecutor(max_workers=self.cfg.tokenize_workers) as pool:
            futures = {pool.submit(self.tokenize_unit, u): u for u in units}
            for done, future in enumerate(as_completed(futures), 1):
                unit_stats[futures[future].uid] = future.result()
                log(f"tokenize: {done}/{len(futures)} units")
        elapsed = time.time() - t0
        new_tokens = sum(s.get("kept_tokens", 0) for s in unit_stats.values())
        log(f"done in {elapsed / 60:.1f} min ({new_tokens / max(elapsed, 1) / 1e6:.1f}M tokens/s incl. units finished "
            "earlier)")
        return unit_stats

    def unit_stats(self, units):
        return {u.uid: json.loads((self.work_dir / "stats" / f"{u.uid}.json").read_text(encoding="utf-8"))
                for u in units}

    def build_manifest(self, units):
        manifest, validation = self.manifest_contents(units)
        save_npy(self.out_dir / "validation.npy", validation)
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        return manifest

    def manifest_contents(self, units):
        cfg = self.cfg
        unit_stats = self.unit_stats(units)
        per_source, shards = {}, []
        for u in units:
            s = unit_stats[u.uid]
            agg = per_source.setdefault(u.source, Counter())
            agg.update({k: v for k, v in s.items() if isinstance(v, int)})
            if s["sequences"]:
                shards.append({"file": f"shards/{u.uid}.npy", "source": u.source, "n_sequences": s["sequences"]})
        validation_stream = [np.load(self.work_dir / "validation" / f"{u.uid}.npy") for u in units]
        validation = pack([p for p in validation_stream if len(p)], cfg.seq_len, self.cls_id,
                          self.sep_id)[: cfg.max_validation_sequences]
        manifest = {
            "format": "packed-uint16-v1",
            "seq_len": cfg.seq_len,
            "dtype": "uint16",
            "vocab_size": self.vocab_size,
            "vocab_fingerprint": self.fingerprint,
            "cls_token_id": self.cls_id,
            "sep_token_id": self.sep_id,
            "total_sequences": sum(s["n_sequences"] for s in shards),
            "total_tokens": sum(s["n_sequences"] for s in shards) * cfg.seq_len,
            "shards": shards,
            "validation": {"file": "validation.npy", "n_sequences": len(validation)},
            "sources": {k: dict(v) for k, v in per_source.items()},
            "config": json.loads(json.dumps(dataclasses.asdict(cfg), default=str)),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return manifest, validation

    def manifest_summary(self, manifest):
        total_tokens = manifest["total_tokens"]
        lines = [f"{'source':<10}{'docs':>13}{'dups':>8}{'short':>8}{'non-ar':>8}{'kept':>13}{'tokens':>10}{'share':>8}"]
        for name, s in manifest["sources"].items():
            docs = max(s["docs"], 1)
            lines.append(f"{name:<10}{s['docs']:>13,}{s.get('duplicate', 0) / docs:>8.2%}"
                         f"{s.get('too_short', 0) / docs:>8.2%}{s.get('non_arabic', 0) / docs:>8.2%}"
                         f"{s.get('kept_docs', 0):>13,}{fmt_tokens(s.get('kept_tokens', 0)):>10}"
                         f"{s.get('sequences', 0) * self.cfg.seq_len / max(total_tokens, 1):>8.1%}")
        lines.append(f"total: {manifest['total_sequences']:,} sequences = {fmt_tokens(total_tokens)} tokens in "
                     f"{len(manifest['shards'])} shards; validation {manifest['validation']['n_sequences']:,} sequences")
        lines.append(f"with 524,288 tokens per step that is {total_tokens / 524_288:,.0f} steps for one epoch")
        return "\n".join(lines)

    def verify(self, manifest):
        cfg, rng = self.cfg, np.random.default_rng(0)
        for shard in rng.choice(manifest["shards"], size=min(5, len(manifest["shards"])), replace=False):
            rows = np.load(self.out_dir / shard["file"], mmap_mode="r")
            if rows.shape != (shard["n_sequences"], cfg.seq_len) or rows.dtype != np.uint16:
                raise AssertionError(f"{shard['file']}: shape {rows.shape}, dtype {rows.dtype}")
            sample = np.asarray(rows[rng.integers(0, len(rows), size=min(64, len(rows)))])
            if not ((sample[:, 0] == self.cls_id).all() and (sample[:, -1] == self.sep_id).all()
                    and sample.max() < self.vocab_size):
                raise AssertionError(f"{shard['file']}: rows are not [CLS] ... [SEP] with valid ids")
        validation = np.load(self.out_dir / "validation.npy", mmap_mode="r")
        if validation.shape[1:] != (cfg.seq_len,) or not (validation[:, 0] == self.cls_id).all():
            raise AssertionError("validation.npy has the wrong layout")
        example = np.load(self.out_dir / manifest["shards"][0]["file"], mmap_mode="r")[0]
        return self.tokenizer.decode(example.tolist(), skip_special_tokens=False)[:600]

    def push(self, api, repo_id, manifest):
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_large_folder(repo_id=repo_id, folder_path=str(self.out_dir), repo_type="dataset", private=True,
                                allow_patterns=["manifest.json", "validation.npy", "shards/*.npy"])
        remote = set(api.list_repo_files(repo_id, repo_type="dataset"))
        missing = [s["file"] for s in manifest["shards"] if s["file"] not in remote]
        log(f"pushed to {repo_id}: {len(remote)} files, missing: {missing or 'none'}")
        if self.cfg.delete_raw_after_push and not missing:
            shutil.rmtree(self.cfg.raw_dir, ignore_errors=True)
            log(f"deleted {self.cfg.raw_dir}")
        return missing
