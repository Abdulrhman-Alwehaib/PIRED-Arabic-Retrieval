import contextlib
import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import CommitOperationDelete, HfApi, snapshot_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from .runlog import log


def checkpoint_step(name):
    return int(name.split("-")[-1])


def rng_states():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_states(states):
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    cuda = states.get("cuda")
    if cuda is not None and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda)


class RandState:
    def __init__(self, device):
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state() if device.type == "cuda" else None

    @contextlib.contextmanager
    def replay(self):
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if self.cuda is not None else []):
            torch.set_rng_state(self.cpu)
            if self.cuda is not None:
                torch.cuda.set_rng_state(self.cuda)
            yield


def write_checkpoint(root, step, model, payload):
    root = Path(root)
    final, tmp = root / f"step-{step:08d}", root / f".tmp-step-{step:08d}"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    torch.save(payload, tmp / "training_state.pt")
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    return final


def read_training_state(ckpt):
    return torch.load(Path(ckpt) / "training_state.pt", map_location="cpu", weights_only=False)


def local_checkpoints(root):
    return sorted(p for p in Path(root).glob("step-*") if (p / "training_state.pt").exists())


def prune_local_checkpoints(root, keep, protected):
    for p in local_checkpoints(root)[:-keep] if keep > 0 else []:
        if p.name not in protected:
            shutil.rmtree(p, ignore_errors=True)


def list_hub_checkpoints(api, repo_id, prefix):
    try:
        entries = api.list_repo_tree(repo_id, path_in_repo=prefix)
        return sorted(Path(e.path).name for e in entries if Path(e.path).name.startswith("step-"))
    except (EntryNotFoundError, RepositoryNotFoundError):
        return []


class HubCheckpointUploader:
    def __init__(self, repo_id, run_name, token, keep_last=3, keep_every_steps=None, squash_history=True):
        self.api = HfApi(token=token)
        self.repo_id, self.prefix = repo_id, f"checkpoints/{run_name}"
        self.keep_last, self.keep_every_steps, self.squash_history = keep_last, keep_every_steps, squash_history
        self.api.create_repo(repo_id, private=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.futures = []
        self.uploaded = set()

    def submit(self, ckpt_dir, pre_decay=False):
        self.futures.append(self.pool.submit(self._upload, Path(ckpt_dir), pre_decay))

    def _upload(self, ckpt_dir, pre_decay):
        t0 = time.time()
        path_in_repo = f"{self.prefix}/pre-decay/{ckpt_dir.name}" if pre_decay else f"{self.prefix}/{ckpt_dir.name}"
        self.api.upload_folder(repo_id=self.repo_id, folder_path=str(ckpt_dir), path_in_repo=path_in_repo,
                               commit_message=f"Add {path_in_repo}")
        if not pre_decay and ckpt_dir.name.startswith("step-"):
            self.uploaded.add(ckpt_dir.name)
            self._prune()
        return f"{path_in_repo} uploaded in {time.time() - t0:.0f}s"

    def hub_checkpoints(self):
        return list_hub_checkpoints(self.api, self.repo_id, self.prefix)

    def not_yet_safe(self, local_names):
        newest = max(self.uploaded, default="")
        return {n for n in local_names if n > newest}

    def _prune(self):
        names = self.hub_checkpoints()
        old = names[: -self.keep_last] if self.keep_last > 0 else names
        doomed = [n for n in old if not self.keep_every_steps or checkpoint_step(n) % self.keep_every_steps != 0]
        if doomed:
            ops = [CommitOperationDelete(path_in_repo=f"{self.prefix}/{n}/") for n in doomed]
            self.api.create_commit(self.repo_id, operations=ops, commit_message=f"Prune {len(doomed)} old checkpoints")
            if self.squash_history:
                self.api.super_squash_history(self.repo_id)

    def report(self, wait=False):
        still_running = []
        for future in self.futures:
            if not future.done() and not wait:
                still_running.append(future)
                continue
            try:
                log(f"[hub] {future.result()}")
            except Exception as e:
                log(f"[hub] upload FAILED: {type(e).__name__}: {e}")
        self.futures = still_running


def download_checkpoint(repo_id, rel, checkpoints_dir, token):
    rel = Path(rel).as_posix()
    staging = Path(checkpoints_dir) / ".hub_download"
    snapshot_download(repo_id, allow_patterns=[f"checkpoints/{rel}/*"], local_dir=staging, token=token)
    if not (staging / "checkpoints" / rel).exists():
        raise FileNotFoundError(f"checkpoints/{rel} is not in {repo_id}")
    target = Path(checkpoints_dir) / rel
    shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging / "checkpoints" / rel), str(target))
    shutil.rmtree(staging, ignore_errors=True)
    log(f"downloaded checkpoints/{rel} from {repo_id}")
    return target


def find_resume_checkpoint(checkpoints_dir, run_name, repo_id, token):
    local = local_checkpoints(Path(checkpoints_dir) / run_name)
    local_step = checkpoint_step(local[-1].name) if local else -1
    if repo_id:
        remote = list_hub_checkpoints(HfApi(token=token), repo_id, f"checkpoints/{run_name}")
        if remote and checkpoint_step(remote[-1]) > local_step:
            return download_checkpoint(repo_id, f"{run_name}/{remote[-1]}", checkpoints_dir, token)
    return local[-1] if local else None
