import dataclasses
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from ..common.checkpoints import HubCheckpointUploader, local_checkpoints
from ..common.device import device_name, free_gpu
from ..common.io import read_json, write_json, write_text
from ..common.runlog import log, read_jsonl
from ..common.text import fmt_hours
from ..encoder.weights import check_reference_outputs, load_encoder, save_reference_outputs
from .training import (BatchLoader, EventLogger, TrainData, TrainState, ValidationSet, build_optimizer,
                       build_scheduler, evaluate_retrieval, find_resume_checkpoint, grad_cache_step, load_checkpoint,
                       save_best, save_checkpoint, train_steps, val_summary)


class Part3Data:
    def __init__(self, stage):
        self.stage = stage
        self.folder = stage.pull_folder(stage.repo_id("pairs"), "dataset", stage.final2)
        self.manifest = read_json(self.folder / "manifest.json")
        self.stats2 = read_json(self.folder / "stats.json")
        part1 = self.folder / "part1_stats.json"
        self.stats1 = read_json(part1) if part1.exists() else {}
        self.n_neg = self.manifest["n_negatives"]
        prefixes = {"query": stage.q_prefix, "passage": stage.p_prefix}
        if self.manifest["tokenizer_fingerprint"] != stage.tokenizer_fingerprint or self.manifest["prefixes"] != prefixes:
            raise AssertionError("Part 2's data was made with another tokenizer or other prefixes")
        self.train = TrainData(stage, self.folder)
        self.val = ValidationSet(stage, self.folder)
        log(f"training rows per source: {self.train.sizes} (total {sum(self.train.sizes.values()):,}); {self.n_neg} "
            f"negatives each; validation: {len(self.val):,} queries, corpus {self.val.n_corpus:,} passages")

    def before_validation(self):
        stage = self.stage
        t0 = time.time()
        before = evaluate_retrieval(stage, stage.encoder, self.val, stage.cfg.train.eval_batch_tokens,
                                    stage.cfg.train.random_pairs)
        stage.timings["validation"] = {"seconds": time.time() - t0, "corpus": self.val.n_corpus,
                                       "queries": len(self.val)}
        log(f"stage-3 model on the validation set (before this stage): {val_summary(before)}; per source nDCG@10: "
            + ", ".join(f"{k} {v['ndcg@10']:.3f}" for k, v in before["per_source"].items())
            + f"; one validation takes {stage.timings['validation']['seconds']:.0f}s here")
        return before


class TrainingRun:
    def __init__(self, stage, data):
        cfg, s, device = stage.cfg, stage.cfg.train, stage.device
        self.stage, self.data, self.settings = stage, data, s
        self.run_dir = Path(cfg.runs_dir) / s.run_name
        self.ckpt_root = Path(cfg.checkpoints_dir) / s.run_name
        if stage.smoke:
            shutil.rmtree(self.ckpt_root, ignore_errors=True)
            (self.run_dir / "train_log.jsonl").unlink(missing_ok=True)
        self.ckpt_root.mkdir(parents=True, exist_ok=True)
        random.seed(s.seed)
        np.random.seed(s.seed)
        torch.manual_seed(s.seed)
        if stage.encoder is not None:
            stage.encoder.to("cpu")
        free_gpu()
        self.model = load_encoder(stage.stage3_dir).to(device)
        if s.compile:
            self.model.compile()
        self.loader = BatchLoader(stage, data.train, data.n_neg, s.batch_size, s.seed, s.chunk_tokens)
        self.total_steps = min(len(self.loader), s.max_steps or len(self.loader))
        self.optimizer = build_optimizer(self.model, s, device)
        self.scheduler = build_scheduler(self.optimizer, self.total_steps, s)
        self.state = TrainState()
        self.logger = EventLogger(self.run_dir / "train_log.jsonl", stage.announce.notifier)
        self.ckpt_repo = stage.repo_id("checkpoints") if cfg.hub.push and stage.token else None
        self.uploader = (HubCheckpointUploader(self.ckpt_repo, s.run_name, stage.token, cfg.hub.keep_last)
                         if self.ckpt_repo else None)

    def describe(self):
        s, n_neg = self.settings, self.data.n_neg
        return "\n".join([
            f"training rows: {sum(self.data.train.sizes.values()):,}; batches per source (batch {s.batch_size}): "
            f"{self.loader.batches_per_source()}; one epoch = {len(self.loader):,} steps"
            + (f"; this run: {self.total_steps} steps (max_steps)" if self.total_steps < len(self.loader) else ""),
            f"loss: InfoNCE over {s.batch_size * (1 + n_neg):,} passages per batch, temperature {s.temperature}, "
            f"reverse direction {s.bidirectional}; lr {s.lr} (warmup {max(1, round(s.warmup_fraction * self.total_steps))}"
            f" steps, linear decay to 0), weight decay {s.weight_decay}, clip {s.max_grad_norm}; GradCache chunks of "
            f"{s.chunk_tokens:,} tokens",
            f"validation every {s.eval_every} steps; checkpoints every "
            + (f"{s.save_every_steps} steps" if s.save_every_steps else f"~{s.save_every_minutes:.0f} min")
            + f" -> {self.ckpt_root} (Hub: {self.ckpt_repo or 'off'}); log -> {self.logger.path}",
        ])

    def estimate(self):
        stage, s, device = self.stage, self.settings, self.stage.device
        model = load_encoder(stage.stage3_dir).to(device).train()
        optimizer = build_optimizer(model, s, device)
        times, tokens = [], []
        for k in range(4):
            batch = self.loader.get(k)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            grad_cache_step(model, batch, s, device)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            if k:
                times.append(time.perf_counter() - t0)
                tokens.append(batch.n_tokens)
        step_s = float(np.mean(times))
        evals = self.total_steps // s.eval_every + 1
        stage.timings["train"] = {"seconds_per_step": step_s, "tokens_per_step": float(np.mean(tokens)),
                                  "batch_size": s.batch_size, "tokens_per_s": float(np.sum(tokens) / np.sum(times)),
                                  "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda"
                                  else 0.0}
        val_s = stage.timings.get("validation", {}).get("seconds", 0.0)
        log(f"{step_s:.2f} s/step ({stage.timings['train']['tokens_per_s']:,.0f} tokens/s, {np.mean(tokens):,.0f} tokens "
            f"per step) on {device_name(device)}; this run: {self.total_steps:,} steps -> "
            f"~{fmt_hours(self.total_steps * step_s)} + {evals} validations x {val_s:.0f}s = "
            f"~{fmt_hours(self.total_steps * step_s + evals * val_s)}")
        del model, optimizer
        free_gpu()

    def resume(self):
        s = self.settings
        latest = find_resume_checkpoint(self.ckpt_root, self.ckpt_repo, s.run_name, self.stage.token) if s.resume else None
        if latest is not None:
            self.state = load_checkpoint(latest, self.model, self.optimizer, self.scheduler, self.loader)
            log(f"resumed from {latest}: step {self.state.step:,}, best validation nDCG@10 so far "
                f"{self.state.best_ndcg:.4f} (step {self.state.best_step}), lr {self.scheduler.get_last_lr()[0]:.2e}")
        else:
            log("no checkpoint of this run: starting from step 0")

    def on_checkpoint(self, state):
        path = save_checkpoint(self.ckpt_root, self.model, self.optimizer, self.scheduler, state, self.loader,
                               self.settings)
        protected = set()
        if self.uploader is not None:
            self.uploader.report()
            self.uploader.submit(path)
            protected = self.uploader.not_yet_safe(p.name for p in local_checkpoints(self.ckpt_root))
        for old in local_checkpoints(self.ckpt_root)[: -self.settings.keep_local_checkpoints]:
            if old.name not in protected:
                shutil.rmtree(old, ignore_errors=True)
        self.logger.log({"step": state.step, "event": "checkpoint", "path": str(path)})

    def on_best(self, state, metrics):
        path = save_best(self.ckpt_root, self.model, state, metrics)
        if self.uploader is not None:
            self.uploader.submit(path)

    def train(self):
        stage, s, state = self.stage, self.settings, self.state
        log(f"training {self.total_steps - state.step:,} steps (step {state.step:,} -> {self.total_steps:,})")
        stage.announce(f"training {s.run_name}: steps {state.step:,} -> {self.total_steps:,}")
        if stage.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            train_steps(stage, self.model, self.optimizer, self.scheduler, self.loader, state,
                        total_steps=self.total_steps, logger=self.logger, val=self.data.val,
                        on_checkpoint=self.on_checkpoint, on_best=self.on_best)
        except KeyboardInterrupt:
            log("interrupted: saving a checkpoint before stopping")
            self.logger.log({"step": state.step, "event": "interrupted"})
            self.on_checkpoint(state)
            raise
        except Exception as e:
            self.logger.log({"step": state.step, "event": "error", "error": f"{type(e).__name__}: {e}"[:1000]})
            raise
        finally:
            if self.uploader is not None:
                log("waiting for checkpoint uploads...")
                self.uploader.report(wait=True)
        self.on_checkpoint(state)
        if self.uploader is not None:
            self.uploader.report(wait=True)
        self.logger.log({"step": state.step, "event": "training finished", "queries_seen": state.queries_seen,
                         "tokens_seen": state.tokens_seen, "elapsed_h": state.elapsed_seconds / 3600,
                         "best_ndcg@10": state.best_ndcg, "best_step": state.best_step})
        self.summary = {"steps": state.step, "queries_seen": state.queries_seen, "tokens_seen": state.tokens_seen,
                        "hours": state.elapsed_seconds / 3600, "best_step": state.best_step,
                        "best_ndcg@10": state.best_ndcg,
                        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if stage.device.type == "cuda" else 0.0}
        records = read_jsonl(self.logger.path)
        lines = ["validation during training:"]
        lines += [f"  step {r['step']:>6}: {val_summary(r)}{'   <- best so far' if r.get('best') else ''}"
                  for r in records if r.get("event") == "validation"]
        rows = [r for r in records if "loss" in r and "event" not in r]
        if rows:
            lines.append(f"train loss {rows[0]['loss']:.3f} -> {rows[-1]['loss']:.3f}, accuracy {rows[0]['acc']:.2f} -> "
                         f"{rows[-1]['acc']:.2f}; best validation nDCG@10 {state.best_ndcg:.4f} at step {state.best_step}")
        log("\n".join(lines))
        return self.summary


def pooling_config(stage, name, which, metrics, total_steps, n_neg):
    s, root = stage.cfg.train, stage.project.root
    stage3 = Path(stage.stage3_dir)
    initialized_from = (stage3.relative_to(root) if stage3.is_relative_to(root) else stage3).as_posix()
    return {"pooling": "mean", "exclude_padding": True, "normalize": "l2", "similarity": "cosine",
            "query_prefix": stage.q_prefix, "passage_prefix": stage.p_prefix, "max_query_tokens": stage.max_q,
            "max_passage_tokens": stage.max_p,
            "note": "limits include the prefix and [CLS]/[SEP]; texts are normalized by the tokenizer only",
            "stage": "supervised with hard negatives (pired.supervised)",
            "checkpoint": name, "chosen_by": which, "step": metrics.get("step"), "total_steps": total_steps,
            "validation": {k: metrics.get(k) for k in ("ndcg@10", "recall@10", "recall@100")},
            "batch_size": s.batch_size, "hard_negatives": n_neg, "temperature": s.temperature,
            "initialized_from": initialized_from}


def export_model(stage, name, source, which, metrics, total_steps, n_neg, check_queries):
    out = Path(stage.cfg.final_dir) / name
    shutil.rmtree(out, ignore_errors=True)
    model = load_encoder(source).to(stage.device).eval()
    model.save_pretrained(out)
    stage.tokenizer.backend_tokenizer.no_truncation()
    stage.tokenizer.save_pretrained(out)
    pooling = pooling_config(stage, name, which, metrics, total_steps, n_neg)
    write_json(pooling, out / "pooling_config.json")
    save_reference_outputs(model, out)
    write_text(f"# {stage.project.name} {name} (stage 4: supervised with hard negatives)"
               f"{' - SMOKE RUN' if stage.smoke else ''}\n\nThe {which} (step {metrics.get('step')} of {total_steps}). "
               f"Arabic encoder ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters), fine-tuned from "
               f"stage 3 on question-style queries with {n_neg} hard negatives each (InfoNCE, in-batch negatives). "
               f"Embedding: mean over real tokens, L2-normalized. Queries start with `{stage.q_prefix}`, passages with "
               f"`{stage.p_prefix}`. The train splits of MIRACL and Mr.TyDi were used. The model code is "
               "`src/pired/encoder` of the project.\n", out / "README.md")
    reloaded = load_encoder(out).to(stage.device).eval()
    diff = (stage.make_embedder(model).encode(check_queries, "query")
            - stage.make_embedder(reloaded).encode(check_queries, "query")).abs().max().item()
    check_reference_outputs(reloaded, out, atol=0.0)
    if diff != 0.0 or read_json(out / "config.json")["architectures"] != ["ArabicEncoderModel"]:
        raise AssertionError(f"{out}: the saved model differs (max |diff| {diff})")
    if AutoTokenizer.from_pretrained(out).backend_tokenizer.truncation is not None:
        raise AssertionError(f"{out}: the saved tokenizer still truncates")
    del model, reloaded
    free_gpu()
    return pooling


def export_models(stage, training, data, tests, before_val):
    cfg, state = stage.cfg, training.state
    export_dir = Path(cfg.final_dir)
    records = read_jsonl(training.logger.path)
    vals = [r for r in records if r.get("event") == "validation"]
    best_metrics = read_json(training.ckpt_root / "best" / "metrics.json")
    last_metrics = next((r for r in reversed(vals) if r["step"] == state.step), vals[-1] if vals else {})
    exports = {cfg.hub.best_name: (training.ckpt_root / "best", "best validation nDCG@10", best_metrics),
               cfg.hub.final_name: (training.ckpt_root / f"step-{state.step:08d}", "last step of training", last_metrics)}
    shutil.rmtree(export_dir, ignore_errors=True)
    check = data.val.q["query"].tolist()[:64]
    for name, (source, which, metrics) in exports.items():
        p = export_model(stage, name, source, which, metrics, state.step, data.n_neg, check)
        log(f"{name}: {which}, step {p['step']} (validation nDCG@10 {(p['validation']['ndcg@10'] or 0):.4f}) -> "
            f"{export_dir / name}")
    if best_metrics.get("step") == state.step:
        log(f"the best checkpoint is the last step: {cfg.hub.best_name} and {cfg.hub.final_name} hold the same weights")
    write_json({"train": training.summary, "tests": tests,
                "timings": {k: stage.timings[k] for k in ("train", "validation") if k in stage.timings},
                "before_validation": before_val, "best": best_metrics, "final": last_metrics, "log": records,
                "settings": dataclasses.asdict(cfg.train)}, training.run_dir / "part3_stats.json")
    run_logs = export_dir / "run_logs"
    run_logs.mkdir(parents=True, exist_ok=True)
    for src, dst in ((training.run_dir / "train_log.jsonl", "train_log.jsonl"),
                     (training.run_dir / "part3_stats.json", "part3_stats.json"),
                     (data.folder / "stats.json", "part2_stats.json"),
                     (data.folder / "part1_stats.json", "part1_stats.json")):
        if Path(src).exists():
            shutil.copyfile(src, run_logs / dst)
    return {"best": best_metrics, "final": last_metrics, "validations": vals}
