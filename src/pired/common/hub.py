from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from .io import folder_files
from .runlog import log


def fetch_dataset_file(repo, path, local_dir, token, revision=None):
    target = Path(local_dir) / path
    if target.exists():
        return target
    return Path(hf_hub_download(repo, path, repo_type="dataset", revision=revision, local_dir=local_dir, token=token))


def push_folder(api, folder, repo_id, repo_type, enabled=True):
    files = folder_files(folder)
    size = sum(p.stat().st_size for p in files) / 1e6
    if not enabled or not api.token:
        log(f"[not uploaded] would upload {len(files)} files ({size:,.1f} MB) from {folder} to the private "
            f"{repo_type} repo {repo_id}")
        return False
    api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, folder_path=str(folder), repo_type=repo_type, private=True)
    log(f"uploaded {len(files)} files ({size:,.1f} MB) to {repo_id}")
    return True


def pull_folder(repo_id, repo_type, local, token, marker="manifest.json"):
    local = Path(local)
    if (local / marker).exists():
        return local
    log(f"{local} not found: downloading {repo_id}")
    snapshot_download(repo_id, repo_type=repo_type, local_dir=local, token=token)
    return local


def missing_files(folder, names):
    return [name for name in names if not (Path(folder) / name).exists()]


def check_on_hub(api, repo_id, files, repo_type="model"):
    missing = sorted(set(files) - set(api.list_repo_files(repo_id, repo_type=repo_type)))
    if missing:
        raise RuntimeError(f"not on the Hub ({repo_id}): {missing[:5]}")
