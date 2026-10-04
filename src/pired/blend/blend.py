import gc
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd, snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from ..common.io import file_sha256, read_json, write_json, write_text
from ..common.runlog import log
from ..encoder.configuration import ArabicEncoderConfig
from ..encoder.embedding import load_tokenizer
from ..encoder.modeling import ArabicEncoderModel
from ..encoder.weights import (MODEL_FILES, check_reference_outputs, config_differences, has_model_files,
                               load_encoder, save_reference_outputs)
from ..tokenizer.fingerprint import check_fingerprint

POOLING_KEYS = ("query_prefix", "passage_prefix", "max_query_tokens", "max_passage_tokens", "pooling", "normalize")


@dataclass
class BlendConfig:
    stage3_dir: Path | None = None
    stage3_repo: str | None = None
    stage4_dir: Path | None = None
    stage4_name: str = "stage4-best"
    model_repo: str | None = None
    tokenizer_fingerprint: str = "b490ba47ebc5a117"
    alphas: tuple[float, ...] = (0.3, 0.5, 0.7)
    prefix: str = "stage4-blend"
    out_dir: Path | None = None
    push: bool = True
    smoke: bool = False


def make_config(project):
    supervised = project.path("supervised_stage_dir") / "final_model"
    return BlendConfig(stage3_dir=project.path("contrastive_weak_stage_dir") / "final_model", stage4_dir=supervised,
                       out_dir=project.stage_dir("supervised_stage_dir") / "final_model", push=not project.smoke,
                       smoke=project.smoke)


def blend_name(cfg, alpha):
    return f"{cfg.prefix}-{alpha:g}"


def blend_state(state3, state4, alpha):
    out = {}
    for k, w4 in state4.items():
        w3 = state3[k]
        out[k] = ((1.0 - alpha) * w3.float() + alpha * w4.float()).to(w4.dtype) if w4.is_floating_point() else w4.clone()
    return out


class Blender:
    def __init__(self, project, cfg, device):
        self.project, self.cfg, self.device = project, cfg, device
        self.s3_dir, self.s4_dir = self.stage3_folder(), self.stage4_folder()
        if not (has_model_files(self.s3_dir) and has_model_files(self.s4_dir)):
            raise FileNotFoundError(f"missing model files in {self.s3_dir} or {self.s4_dir}")
        self.state3 = load_file(str(self.s3_dir / "model.safetensors"))
        self.state4 = load_file(str(self.s4_dir / "model.safetensors"))
        self.sha = {"stage 3": file_sha256(self.s3_dir / "model.safetensors"),
                    "stage 4": file_sha256(self.s4_dir / "model.safetensors")}
        self.model_config = ArabicEncoderConfig.from_pretrained(self.s4_dir)
        if diff := config_differences(ArabicEncoderConfig.from_pretrained(self.s3_dir), self.model_config):
            raise AssertionError(f"the two models have different architectures: {diff}")
        if self.state3.keys() != self.state4.keys() or any(self.state3[k].shape != self.state4[k].shape
                                                           for k in self.state4):
            raise AssertionError("the two models have different tensors")
        self.pool3, self.pool4 = read_json(self.s3_dir / "pooling_config.json"), read_json(self.s4_dir / "pooling_config.json")
        if not all(self.pool3[k] == self.pool4[k] for k in POOLING_KEYS):
            raise AssertionError({k: (self.pool3.get(k), self.pool4.get(k)) for k in POOLING_KEYS})
        for folder in (self.s3_dir, self.s4_dir):
            check_fingerprint(AutoTokenizer.from_pretrained(folder), cfg.tokenizer_fingerprint, str(folder))
        self.tokenizer = load_tokenizer(self.s4_dir)
        log(f"stage 3: {self.s3_dir} (weights {self.sha['stage 3'][:12]}); stage 4: {self.s4_dir} (weights "
            f"{self.sha['stage 4'][:12]}, step {self.pool4.get('step')}); same architecture "
            f"({self.model_config.num_hidden_layers} layers, hidden {self.model_config.hidden_size}), the same "
            f"{len(self.state4)} tensors, the same tokenizer ({cfg.tokenizer_fingerprint}), prefixes "
            f"{self.pool4['query_prefix']!r} / {self.pool4['passage_prefix']!r}, limits {self.pool4['max_query_tokens']} / "
            f"{self.pool4['max_passage_tokens']} tokens")

    def stage3_folder(self):
        cfg = self.cfg
        if not has_model_files(cfg.stage3_dir):
            repo = self.project.repo_id("contrastive-weak", cfg.stage3_repo)
            snapshot_download(repo, allow_patterns=list(MODEL_FILES), local_dir=cfg.stage3_dir,
                              token=self.project.hf_token)
            log(f"stage 3: downloaded from {repo}")
        return Path(cfg.stage3_dir)

    def stage4_folder(self):
        cfg = self.cfg
        local = Path(cfg.stage4_dir) / cfg.stage4_name
        if not has_model_files(local):
            repo = self.project.repo_id("supervised", cfg.model_repo)
            snapshot_download(repo, allow_patterns=[f"{cfg.stage4_name}/{f}" for f in MODEL_FILES],
                              local_dir=cfg.stage4_dir, token=self.project.hf_token)
            log(f"{cfg.stage4_name}: downloaded from {repo}")
        return local

    def build(self, alpha):
        cfg, device = self.cfg, self.device
        t0 = time.time()
        name = blend_name(cfg, alpha)
        out = Path(cfg.out_dir) / name
        shutil.rmtree(out, ignore_errors=True)
        state = blend_state(self.state3, self.state4, alpha)
        model = ArabicEncoderModel(self.model_config)
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        write_json({**self.pool4, "checkpoint": name, "step": f"blend alpha={alpha:g}",
                    "chosen_by": "one of the blend stage's mixes; the evaluation stage compares them on mteb",
                    "stage": "mix of stage 3 and stage 4 (pired.blend)",
                    "blend": {"alpha": alpha, "stage3": {"folder": str(self.s3_dir), "sha256": self.sha["stage 3"]},
                              "stage4": {"folder": str(self.s4_dir), "sha256": self.sha["stage 4"]}},
                    "validation": None}, out / "pooling_config.json")
        save_reference_outputs(model, out)
        write_text(f"# {self.project.name} {name}{' - SMOKE RUN' if cfg.smoke else ''}\n\n"
                   f"A weight mix of the stage-3 model and {cfg.stage4_name}: w = {1 - alpha:g} x stage 3 + {alpha:g} x "
                   f"{cfg.stage4_name}. Embedding: mean over real tokens, L2-normalized. Queries start with "
                   f"`{self.pool4['query_prefix']}`, passages with `{self.pool4['passage_prefix']}`. The model code is "
                   "`src/pired/encoder` of the project.\n", out / "README.md")
        reloaded = load_encoder(out)
        if not all(torch.equal(reloaded.state_dict()[k], state[k]) for k in state):
            raise AssertionError(f"{out}: saved weights differ")
        check_reference_outputs(reloaded.to(device).eval(), out, atol=0.0)
        if not has_model_files(out) or read_json(out / "config.json")["architectures"] != ["ArabicEncoderModel"]:
            raise AssertionError(f"{out}: incomplete folder")
        if AutoTokenizer.from_pretrained(out).backend_tokenizer.truncation is not None:
            raise AssertionError(f"{out}: the saved tokenizer still truncates")
        saved = {"folder": str(out), "sha256": file_sha256(out / "model.safetensors")}
        del model, reloaded, state
        gc.collect()
        log(f"{name}: {1 - alpha:g} x stage 3 + {alpha:g} x {cfg.stage4_name} -> {out} (weights {saved['sha256'][:12]}; "
            f"reloaded: same tensors, reference outputs reproduced; {time.time() - t0:.0f}s)")
        return saved

    def build_all(self):
        return {alpha: self.build(alpha) for alpha in self.cfg.alphas}

    def upload(self, saved):
        cfg, api = self.cfg, self.project.api
        record = Path(cfg.out_dir) / "blend_mixes.json"
        write_json({"alphas": list(cfg.alphas), "sources": self.sha,
                    "mixes": {blend_name(cfg, a): v for a, v in saved.items()},
                    "date": time.strftime("%Y-%m-%d %H:%M")}, record)
        repo = self.project.repo_id("supervised", cfg.model_repo)
        files = {f"{blend_name(cfg, a)}/{p.relative_to(v['folder']).as_posix()}": p for a, v in saved.items()
                 for p in sorted(Path(v["folder"]).rglob("*")) if p.is_file()}
        files["run_logs/blend_mixes.json"] = record
        names = ", ".join(f"{blend_name(cfg, a)}/" for a in cfg.alphas)
        if not (cfg.push and self.project.hf_token):
            log(f"[not uploaded: {'SMOKE' if cfg.smoke else 'push off or no HF_TOKEN'}] would upload {len(files)} files to "
                f"{repo}: {names}")
            return
        api.create_commit(repo, operations=[CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v))
                                            for k, v in files.items()],
                          commit_message=f"Add the mixes {', '.join(blend_name(cfg, a) for a in cfg.alphas)}")
        missing = sorted(set(files) - set(api.list_repo_files(repo)))
        if missing:
            raise RuntimeError(f"not on the Hub: {missing}")
        log(f"uploaded {len(files)} files to {repo}: {names}")
