import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(json.dumps(obj, ensure_ascii=False, indent=1, default=str).encode("utf-8"))
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def write_parquet(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def replace_dir(tmp, final, attempts=25):
    tmp, final = Path(tmp), Path(final)
    for k in range(attempts):
        shutil.rmtree(final, ignore_errors=True)
        try:
            tmp.rename(final)
            return final
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(min(0.25 * (k + 1), 3.0))
    return final


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def folder_files(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and ".cache" not in p.parts)


def folder_size_mb(folder):
    return sum(p.stat().st_size for p in folder_files(folder)) / 1e6
