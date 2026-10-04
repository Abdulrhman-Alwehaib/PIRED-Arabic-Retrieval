import dataclasses
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from ..common.device import autocast_ctx
from ..common.notify import make_notifier
from ..common.runlog import JsonlLogger, log
from ..encoder.configuration import ArabicEncoderConfig
from ..encoder.modeling import ArabicEncoderForMaskedLM
from ..encoder.weights import save_reference_outputs
from .checkpoints import (TrainState, find_init_checkpoint, find_resume_checkpoint, make_uploader,
                          load_checkpoint, local_checkpoints, prune_local_checkpoints, save_checkpoint)
from .collator import MLMCollator
from .data import PretrainingData
from .schedule import build_optimizer, build_scheduler, decay_start_step, lr_factor


def unique_param_count(params):
    seen = {}
    for p in params:
        seen[id(p)] = p.numel()
    return sum(seen.values())


def parameter_breakdown(model):
    tied = model.model.embeddings.tok_embeddings.weight
    body = [*model.model.layers.parameters(), *model.model.final_norm.parameters()]
    parts = {
        "embeddings (token table + LayerNorm)": unique_param_count(model.model.embeddings.parameters()),
        "transformer body (layers + final LayerNorm)": unique_param_count(body),
        "MLM head (dense + LayerNorm + decoder bias; decoder weight tied)":
            unique_param_count(p for p in model.head.parameters() if p is not tied),
    }
    parts["total (with MLM head)"] = unique_param_count(model.parameters())
    parts["encoder only (what stages 3-5 use)"] = unique_param_count(model.model.parameters())
    if sum(list(parts.values())[:3]) != parts["total (with MLM head)"]:
        raise AssertionError("the parameter groups do not add up")
    return parts


def format_breakdown(model):
    c = model.config
    lines = [f"vocab={c.vocab_size} hidden={c.hidden_size} layers={c.num_hidden_layers} heads={c.num_attention_heads} "
             f"head_dim={c.head_dim} ffn(GeGLU)={c.intermediate_size}"]
    lines += [f"  {name:<66}{n:>14,}  ({n / 1e6:7.2f}M)" for name, n in parameter_breakdown(model).items()]
    return "\n".join(lines)


@torch.no_grad()
def evaluate_mlm(model, sequences, collator, batch_size, device):
    collator.generator.manual_seed(0)
    model.eval()
    loss_sum, n_targets = 0.0, 0
    for i in range(0, len(sequences), batch_size):
        batch = {k: v.to(device) for k, v in collator(sequences[i: i + batch_size]).items()}
        with autocast_ctx(device):
            loss = model(**batch).loss
        k = int((batch["labels"] != -100).sum())
        loss_sum, n_targets = loss_sum + loss.item() * k, n_targets + k
    return loss_sum / max(n_targets, 1)


def train_steps(model, optimizer, scheduler, data, collator, state, *, num_steps, grad_accum_steps, max_grad_norm,
                device, log_every=10, logger=None, save_every=None, on_checkpoint=None, pre_decay_step=None,
                on_pre_decay=None, evaluate=None, eval_every=None, record_losses=False):
    model.train()
    losses = []
    window_loss, window_steps, window_tokens = torch.zeros((), device=device), 0, 0
    window_start = time.perf_counter()
    grad_norm = torch.zeros(())
    for _ in range(num_steps):
        micro_batches = [collator(next(data)) for _ in range(grad_accum_steps)]
        n_targets = sum(int((b["labels"] != -100).sum()) for b in micro_batches)
        n_tokens = sum(int(b["attention_mask"].sum()) if "attention_mask" in b else b["input_ids"].numel()
                       for b in micro_batches)
        optimizer.zero_grad(set_to_none=True)
        step_loss = torch.zeros((), device=device)
        for b in micro_batches:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with autocast_ctx(device):
                loss = model(**b, num_items_in_batch=n_targets).loss
            loss.backward()
            step_loss += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        state.step += 1
        state.tokens_seen += n_tokens
        window_loss += step_loss
        window_steps += 1
        window_tokens += n_tokens
        if record_losses:
            losses.append(step_loss.item())
        if logger is not None and state.step % log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            state.elapsed_seconds += now - window_start
            logger.log({
                "step": state.step,
                "loss": (window_loss / window_steps).item(),
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "tokens_per_sec": window_tokens / (now - window_start),
                "tokens_seen": state.tokens_seen,
                "elapsed_h": state.elapsed_seconds / 3600,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
            })
            window_loss.zero_()
            window_steps, window_tokens, window_start = 0, 0, time.perf_counter()
        if evaluate is not None and eval_every and state.step % eval_every == 0:
            paused = time.perf_counter()
            val_loss = evaluate()
            if logger is not None:
                logger.log({"step": state.step, "val_loss": val_loss})
            model.train()
            window_start += time.perf_counter() - paused
        if on_pre_decay is not None and state.step == pre_decay_step:
            paused = time.perf_counter()
            on_pre_decay(state)
            window_start += time.perf_counter() - paused
        if save_every and on_checkpoint is not None and state.step % save_every == 0:
            paused = time.perf_counter()
            on_checkpoint(state)
            window_start += time.perf_counter() - paused
    return losses


def load_tokenizer(cfg, project):
    if (Path(cfg.tokenizer_dir) / "tokenizer.json").exists():
        return AutoTokenizer.from_pretrained(cfg.tokenizer_dir)
    repo = project.repo_id("tokenizer", cfg.tokenizer_repo)
    tokenizer = AutoTokenizer.from_pretrained(repo, token=project.hf_token)
    tokenizer.save_pretrained(cfg.tokenizer_dir)
    return tokenizer


def make_model_config(settings, tokenizer):
    return ArabicEncoderConfig(vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id,
                               cls_token_id=tokenizer.cls_token_id, sep_token_id=tokenizer.sep_token_id,
                               mask_token_id=tokenizer.mask_token_id, **dataclasses.asdict(settings))


def data_repo_id(cfg, project):
    if cfg.data.repo_id:
        return cfg.data.repo_id
    return project.repo_id("pretraining-data") if project.hf_token else None


def checkpoint_repo_id(cfg, project):
    if not (cfg.hub.push_checkpoints and project.hf_token):
        return None
    return project.repo_id("mlm-checkpoints", cfg.hub.checkpoint_repo)


class PretrainingRun:
    def __init__(self, cfg, project, device):
        if device.type != "cuda" and not cfg.smoke:
            raise RuntimeError("pretraining needs a CUDA GPU with bf16")
        s = cfg.train
        self.cfg, self.project, self.device, self.settings = cfg, project, device, s
        self.run_dir = Path(cfg.runs_dir) / s.run_name
        self.ckpt_root = Path(cfg.checkpoints_dir) / s.run_name
        self.ckpt_root.mkdir(parents=True, exist_ok=True)
        random.seed(s.seed)
        np.random.seed(s.seed)
        torch.manual_seed(s.seed)
        self.tokenizer = load_tokenizer(cfg, project)
        self.model = ArabicEncoderForMaskedLM(make_model_config(cfg.model, self.tokenizer)).to(device)
        if s.compile:
            self.model.model.compile()
        self.optimizer = build_optimizer(self.model, s, device)
        self.scheduler = build_scheduler(self.optimizer, s)
        self.collator = MLMCollator.from_tokenizer(self.tokenizer, cfg.mlm, seed=s.seed)
        self.data = PretrainingData(self.tokenizer, s.seq_len, s.micro_batch_size, s.seed, cfg.data,
                                    data_repo_id(cfg, project), project.hf_token)
        self.val_rows = self.data.validation_sequences(s.eval_sequences)
        self.eval_collator = MLMCollator.from_tokenizer(self.tokenizer, cfg.mlm)
        self.state = TrainState()
        notifier = make_notifier(f"MLM {s.run_name}", cfg.notify.telegram)
        self.logger = JsonlLogger(self.run_dir / "train_log.jsonl", notifier=notifier)
        self.repo_id = checkpoint_repo_id(cfg, project)
        self.uploader = make_uploader(self.repo_id, s.run_name, cfg.hub, project.hf_token) if self.repo_id else None
        self.pre_decay_step = decay_start_step(s) if s.lr_schedule == "wsd" else None

    def describe(self):
        s = self.settings
        tokens_per_step = s.micro_batch_size * s.grad_accum_steps * s.seq_len
        lines = [format_breakdown(self.model),
                 f"data: {self.data.describe()}; validation rows: {len(self.val_rows):,}",
                 f"plan: {s.total_steps:,} steps x {tokens_per_step:,} tokens = "
                 f"{s.total_steps * tokens_per_step / 1e9:.1f}B tokens = "
                 f"{s.total_steps * tokens_per_step / (self.data.total * s.seq_len):.2f} epochs"]
        if self.pre_decay_step is not None:
            lines.append(f"lr: warmup {s.warmup_steps:,} steps, {s.lr:.1e} until step {self.pre_decay_step:,} "
                         f"(pre-decay checkpoint), then 1-sqrt decay to {s.lr * s.min_lr_ratio:.1e} at step "
                         f"{s.total_steps:,}")
        else:
            lines.append(f"lr: warmup {s.warmup_steps:,} steps, then {s.lr_schedule} decay to {s.lr * s.min_lr_ratio:.1e}")
        lines.append(f"checkpoints: {self.ckpt_root}  hub: {self.repo_id or 'off'}  log: {self.logger.path}")
        return "\n".join(lines)

    def evaluate(self):
        return evaluate_mlm(self.model, self.val_rows, self.eval_collator, self.settings.micro_batch_size, self.device)

    def resume(self):
        s = self.settings
        parts = (self.model, self.optimizer, self.scheduler, self.data, self.collator)
        latest = find_resume_checkpoint(self.cfg.checkpoints_dir, s.run_name, self.repo_id,
                                        self.project.hf_token) if s.resume else None
        if latest is not None:
            self.state = load_checkpoint(latest, *parts, s)
            log(f"resumed from {latest.name}: step {self.state.step:,}, {self.state.tokens_seen / 1e9:.2f}B tokens, "
                f"lr {self.scheduler.get_last_lr()[0]:.2e}")
        elif s.init_from:
            start = find_init_checkpoint(s.init_from, self.cfg.checkpoints_dir, self.repo_id, self.project.hf_token)
            self.state = load_checkpoint(start, *parts, s, allow_new_data=True)
            if self.pre_decay_step is not None and self.state.step > self.pre_decay_step:
                raise ValueError(f"{s.init_from} is at step {self.state.step:,}, past this run's decay start "
                                 f"({self.pre_decay_step:,}); raise total_steps")
            saved_lr = self.scheduler.get_last_lr()[0]
            planned_lr = self.scheduler.base_lrs[0] * lr_factor(self.state.step, s)
            if saved_lr < 0.99 * planned_lr:
                log(f"WARNING: {s.init_from} was saved during a decay (lr {saved_lr:.2e}); this run continues at "
                    f"{planned_lr:.2e}")
            log(f"started from {s.init_from}: step {self.state.step:,}, {self.state.tokens_seen / 1e9:.2f}B tokens")
        else:
            log("no checkpoint found: starting from step 0")

    def on_checkpoint(self, state):
        path = save_checkpoint(self.ckpt_root, self.model, self.optimizer, self.scheduler, self.data, self.collator,
                               state, self.settings)
        if self.uploader is not None:
            self.uploader.report()
            self.uploader.submit(path)
            protected = self.uploader.not_yet_safe(p.name for p in local_checkpoints(self.ckpt_root))
        else:
            protected = set()
        prune_local_checkpoints(self.ckpt_root, self.settings.keep_local_checkpoints, protected)
        self.logger.log({"step": state.step, "event": "checkpoint", "path": str(path)})

    def on_pre_decay(self, state):
        path = save_checkpoint(self.ckpt_root / "pre-decay", self.model, self.optimizer, self.scheduler, self.data,
                               self.collator, state, self.settings)
        if self.uploader is not None:
            self.uploader.submit(path, pre_decay=True)
        self.logger.log({"step": state.step, "event": "pre-decay checkpoint", "path": str(path)})

    def train(self):
        s = self.settings
        remaining = s.total_steps - self.state.step
        log(f"training {remaining:,} steps (step {self.state.step:,} -> {s.total_steps:,})")
        try:
            train_steps(self.model, self.optimizer, self.scheduler, self.data, self.collator, self.state,
                        num_steps=remaining, grad_accum_steps=s.grad_accum_steps, max_grad_norm=s.max_grad_norm,
                        device=self.device, log_every=s.log_every, logger=self.logger,
                        save_every=s.save_every_steps, on_checkpoint=self.on_checkpoint,
                        pre_decay_step=self.pre_decay_step, on_pre_decay=self.on_pre_decay,
                        evaluate=self.evaluate, eval_every=s.eval_every_steps)
        except KeyboardInterrupt:
            log("interrupted: saving a checkpoint before stopping")
            self.logger.log({"step": self.state.step, "event": "interrupted"})
            self.on_checkpoint(self.state)
            raise
        except Exception as e:
            self.logger.log({"step": self.state.step, "event": "error", "error": f"{type(e).__name__}: {e}"[:1000]})
            raise
        finally:
            if self.uploader is not None:
                log("waiting for checkpoint uploads...")
                self.uploader.report(wait=True)
        if self.state.step % s.save_every_steps:
            self.on_checkpoint(self.state)
            if self.uploader is not None:
                self.uploader.report(wait=True)
        self.logger.log({"step": self.state.step, "event": "training finished", "tokens_seen": self.state.tokens_seen,
                         "elapsed_h": self.state.elapsed_seconds / 3600})

    def save_final(self):
        final_dir = Path(self.cfg.final_model_dir)
        mlm_dir, encoder_dir = save_final_model(self.model, self.tokenizer, final_dir)
        log(f"MLM model -> {mlm_dir} {sorted(p.name for p in mlm_dir.iterdir())}")
        log(f"encoder   -> {encoder_dir} {sorted(p.name for p in encoder_dir.iterdir())}")
        self.logger.log({"step": self.state.step, "event": "final model saved", "path": str(final_dir)})

    def close(self):
        self.data.close()


def save_final_model(model, tokenizer, final_dir):
    mlm_dir, encoder_dir = Path(final_dir) / "mlm", Path(final_dir) / "encoder"
    model.save_pretrained(mlm_dir)
    tokenizer.save_pretrained(mlm_dir)
    save_reference_outputs(model, mlm_dir)
    model.save_encoder(encoder_dir)
    tokenizer.save_pretrained(encoder_dir)
    save_reference_outputs(model.model, encoder_dir)
    return mlm_dir, encoder_dir
