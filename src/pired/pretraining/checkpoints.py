import dataclasses
from pathlib import Path

from ..common.checkpoints import (HubCheckpointUploader, checkpoint_step, download_checkpoint,
                                  find_resume_checkpoint, list_hub_checkpoints, local_checkpoints,
                                  prune_local_checkpoints, read_training_state, restore_rng_states, rng_states,
                                  write_checkpoint)
from ..common.runlog import log
from ..encoder.weights import load_weights_strict

__all__ = ["HubCheckpointUploader", "TrainState", "checkpoint_step", "download_checkpoint", "find_init_checkpoint",
           "find_resume_checkpoint", "list_hub_checkpoints", "load_checkpoint", "local_checkpoints",
           "make_uploader", "prune_local_checkpoints", "save_checkpoint"]


@dataclasses.dataclass
class TrainState:
    step: int = 0
    tokens_seen: int = 0
    elapsed_seconds: float = 0.0


def changed_settings(saved, settings):
    current = dataclasses.asdict(settings)
    return {k: (v, current[k]) for k, v in saved.items()
            if k in current and current[k] != v and not (isinstance(v, list) and tuple(v) == current[k])}


def save_checkpoint(root, model, optimizer, scheduler, data, collator, state, settings):
    return write_checkpoint(root, state.step, model, {
        "state": dataclasses.asdict(state),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "data": data.state_dict(),
        "collator": collator.state_dict(),
        "rng": rng_states(),
        "train_settings": dataclasses.asdict(settings),
    })


def load_checkpoint(ckpt_dir, model, optimizer, scheduler, data, collator, settings=None, allow_new_data=False):
    load_weights_strict(model, ckpt_dir)
    ts = read_training_state(ckpt_dir)
    optimizer.load_state_dict(ts["optimizer"])
    scheduler.load_state_dict(ts["scheduler"])
    if allow_new_data and ts["data"].get("total_sequences") != getattr(data, "total", None):
        log("the dataset changed since this checkpoint: reading it from its first row")
    else:
        data.load_state_dict(ts["data"])
    collator.load_state_dict(ts["collator"])
    restore_rng_states(ts["rng"])
    if settings is not None and (changed := changed_settings(ts["train_settings"], settings)):
        log(f"WARNING: settings changed since this checkpoint (saved, now): {changed}")
    return TrainState(**ts["state"])


def make_uploader(repo_id, run_name, hub, token):
    return HubCheckpointUploader(repo_id, run_name, token, hub.keep_last, hub.keep_every_steps, hub.squash_history)


def find_init_checkpoint(rel, checkpoints_dir, repo_id, token):
    local = Path(checkpoints_dir) / rel
    if (local / "training_state.pt").exists():
        return local
    if not repo_id:
        raise FileNotFoundError(f"{local} is missing and Hub checkpoints are off")
    return download_checkpoint(repo_id, rel, checkpoints_dir, token)
