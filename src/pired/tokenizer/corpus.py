import dataclasses
import hashlib
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi

from ..common.runlog import log


def list_parquet_shards(repo_id, prefix, token=None):
    files = HfApi(token=token).list_repo_files(repo_id, repo_type="dataset")
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))


def is_heldout(text, every):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % every == 0


def fetch_shard(source, shard_path, quota, part_stem, cfg):
    from datasets import load_dataset

    done_file = Path(f"{part_stem}.done.json")
    if done_file.exists():
        return json.loads(done_file.read_text(encoding="utf-8"))
    rows = load_dataset("parquet", data_files=[f"hf://datasets/{source.repo_id}/{shard_path}"],
                        split="train", streaming=True)
    n_train = n_heldout = 0
    with open(f"{part_stem}.train.jsonl", "w", encoding="utf-8") as f_train, \
            open(f"{part_stem}.heldout.jsonl", "w", encoding="utf-8") as f_heldout:
        for row in rows:
            text = row.get(source.text_field) or ""
            if len(text.strip()) < cfg.min_doc_chars:
                continue
            line = json.dumps({"text": text, "source": source.name}, ensure_ascii=False) + "\n"
            if is_heldout(text, cfg.heldout_every):
                f_heldout.write(line)
                n_heldout += 1
            else:
                f_train.write(line)
                n_train += 1
                if n_train >= quota:
                    break
    stats = {"shard": shard_path, "train": n_train, "heldout": n_heldout}
    done_file.write_text(json.dumps(stats), encoding="utf-8")
    return stats


def corpus_settings(cfg):
    return {"sources": [dataclasses.asdict(s) for s in cfg.sources], "heldout_every": cfg.heldout_every,
            "min_doc_chars": cfg.min_doc_chars, "seed": cfg.seed}


def build_corpus(cfg, token=None):
    corpus_dir = Path(cfg.corpus_dir)
    parts_dir = corpus_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = corpus_dir / "manifest.json"
    wanted = corpus_settings(cfg)
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest["settings"] == wanted:
            log(f"corpus already built: {manifest['counts']}")
            return manifest
    jobs = []
    for source in cfg.sources:
        shards = list_parquet_shards(source.repo_id, source.shard_prefix, token)
        chosen = sorted(random.Random(f"{cfg.seed}-{source.name}").sample(shards, min(source.n_shards, len(shards))))
        quota = math.ceil(source.n_train_docs / len(chosen))
        for shard_path in chosen:
            stem = parts_dir / f"{source.name}-{Path(shard_path).stem}-q{quota}-h{cfg.heldout_every}-m{cfg.min_doc_chars}"
            jobs.append((source, stem, shard_path, quota))
        log(f"{source.name}: {len(chosen)} of {len(shards)} shards, {quota:,} docs each")
    with ThreadPoolExecutor(max_workers=cfg.download_workers) as pool:
        futures = [pool.submit(fetch_shard, src, path, quota, stem, cfg) for src, stem, path, quota in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            log(f"shards: {done}/{len(futures)}")
    counts = {}
    with open(corpus_dir / "train.jsonl", "w", encoding="utf-8") as f_train, \
            open(corpus_dir / "heldout.jsonl", "w", encoding="utf-8") as f_heldout:
        for source in cfg.sources:
            c = {"train_docs": 0, "train_words": 0, "heldout_docs": 0, "heldout_words": 0}
            for src, stem, _, _ in jobs:
                if src is not source:
                    continue
                for split, out, limit in (("train", f_train, source.n_train_docs),
                                          ("heldout", f_heldout, source.n_heldout_docs)):
                    with open(f"{stem}.{split}.jsonl", encoding="utf-8") as f_part:
                        for line in f_part:
                            if c[f"{split}_docs"] >= limit:
                                break
                            out.write(line)
                            c[f"{split}_docs"] += 1
                            c[f"{split}_words"] += len(json.loads(line)["text"].split())
            counts[source.name] = c
    manifest = {"settings": wanted, "counts": counts}
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_manifest(cfg):
    return json.loads((Path(cfg.corpus_dir) / "manifest.json").read_text(encoding="utf-8"))


def iter_corpus_batches(path, batch_size=1_000):
    batch = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            batch.append(json.loads(line)["text"])
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def load_heldout(cfg):
    with open(Path(cfg.corpus_dir) / "heldout.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
