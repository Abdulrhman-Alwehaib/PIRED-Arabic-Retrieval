import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from ..common.checkpoints import HubCheckpointUploader, find_resume_checkpoint, local_checkpoints
from ..common.device import free_gpu
from ..common.hub import push_folder
from ..common.io import read_json, write_json, write_text
from ..common.notify import make_notifier
from ..common.runlog import JsonlLogger, log, read_jsonl
from ..common.text import md_table
from ..encoder.modeling import ArabicEncoderModel
from ..encoder.weights import check_reference_outputs, load_encoder_without_head, save_reference_outputs
from ..evaluation.mteb_runner import baseline_encoder, get_tasks, ours_encoder, run_mteb_tasks
from .training import (BatchLoader, PairData, TrainState, ValidationSet, build_optimizer, build_scheduler,
                       ensure_final_data, evaluate_pairs, load_checkpoint, pair_metrics, save_checkpoint,
                       train_steps, val_summary)


def run_dir(stage):
    return Path(stage.cfg.runs_dir) / stage.cfg.train.run_name


def eval_results_path(stage):
    return run_dir(stage) / "eval_results.json"


def load_eval_results(stage):
    path = eval_results_path(stage)
    return read_json(path) if path.exists() else {}


def save_eval_results(stage, results):
    path = eval_results_path(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")


def prepare_training_data(stage):
    if stage.tokenizer is None:
        stage.load_encoder()
    ensure_final_data(stage)
    return PairData(stage, "train"), ValidationSet(stage)


class TrainingRun:
    def __init__(self, stage, train_data, val):
        cfg, s, device = stage.cfg, stage.cfg.train, stage.device
        self.stage, self.settings, self.train_data, self.val = stage, s, train_data, val
        self.run_dir = run_dir(stage)
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
        self.model = load_encoder_without_head(stage.mlm_dir)[0].to(device)
        if s.compile:
            self.model.compile()
        self.loader = BatchLoader(stage, train_data, s.batch_size, s.seed, s.chunk_tokens)
        self.total_steps = min(len(self.loader), s.max_steps or len(self.loader))
        self.optimizer = build_optimizer(self.model, s, device)
        self.scheduler = build_scheduler(self.optimizer, self.total_steps, s)
        self.state = TrainState()
        self.logger = JsonlLogger(self.run_dir / "train_log.jsonl",
                                  notifier=make_notifier(f"04 contrastive {s.run_name}", cfg.notify.telegram))
        self.ckpt_repo = stage.repo_id("checkpoints") if cfg.hub.push and stage.token else None
        self.uploader = (HubCheckpointUploader(self.ckpt_repo, s.run_name, stage.token, cfg.hub.keep_last)
                         if self.ckpt_repo else None)

    def describe(self):
        s, data = self.settings, self.train_data
        pairs = sum(data.sizes.values())
        return "\n".join([
            f"train pairs: {pairs:,} {data.sizes}",
            f"batches per source (batch {s.batch_size:,}): {self.loader.batches_per_source()}; one pass = "
            f"{len(self.loader):,} steps" + (f"; this run: {self.total_steps} steps (max_steps)"
                                             if self.total_steps < len(self.loader) else ""),
            f"loss: InfoNCE, temperature {s.temperature}, bidirectional {s.bidirectional}; lr {s.lr} (warmup "
            f"{max(1, round(s.warmup_fraction * self.total_steps))} steps, linear decay to 0), weight decay "
            f"{s.weight_decay}, clip {s.max_grad_norm}; GradCache chunks of {s.chunk_tokens:,} tokens",
            f"validation every {s.eval_every} steps on {len(self.val):,} pairs; checkpoints every {s.save_every} steps "
            f"-> {self.ckpt_root} (Hub: {self.ckpt_repo or 'off'}); log -> {self.logger.path}",
        ])

    def resume(self):
        s = self.settings
        latest = find_resume_checkpoint(self.stage.cfg.checkpoints_dir, s.run_name, self.ckpt_repo,
                                        self.stage.token) if s.resume else None
        if latest is not None:
            self.state = load_checkpoint(latest, self.model, self.optimizer, self.scheduler, self.loader)
            log(f"resumed from {latest}: step {self.state.step:,}, {self.state.pairs_seen:,} pairs seen, lr "
                f"{self.scheduler.get_last_lr()[0]:.2e}")
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

    def train(self):
        s, device, state = self.settings, self.stage.device, self.state
        log(f"training {self.total_steps - state.step:,} steps (step {state.step:,} -> {self.total_steps:,})")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            train_steps(self.stage, self.model, self.optimizer, self.scheduler, self.loader, state,
                        total_steps=self.total_steps, logger=self.logger, val=self.val,
                        on_checkpoint=self.on_checkpoint)
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
        if state.step % s.save_every:
            self.on_checkpoint(state)
            if self.uploader is not None:
                self.uploader.report(wait=True)
        self.logger.log({"step": state.step, "event": "training finished", "pairs_seen": state.pairs_seen,
                         "tokens_seen": state.tokens_seen, "elapsed_h": state.elapsed_seconds / 3600})
        write_json({"seconds": state.elapsed_seconds, "steps": state.step, "pairs_seen": state.pairs_seen,
                    "tokens": state.tokens_seen, "batch_size": s.batch_size,
                    "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0},
                   self.run_dir / "train_summary.json")
        return training_summary(self.stage)

    def export(self):
        return export_final_model(self.stage, self.model, self.state, self.val)


def training_summary(stage):
    records = read_jsonl(run_dir(stage) / "train_log.jsonl")
    lines = ["validation during training:"]
    lines += [f"  step {r['step']:>6}: {val_summary(r)}" for r in records if r.get("event") == "validation"]
    rows = [r for r in records if "loss" in r and "event" not in r]
    summary_file = run_dir(stage) / "train_summary.json"
    if rows and summary_file.exists():
        t = read_json(summary_file)
        lines.append(f"train loss {rows[0]['loss']:.3f} -> {rows[-1]['loss']:.3f}, in-batch accuracy "
                     f"{rows[0]['acc_qp']:.2f} -> {rows[-1]['acc_qp']:.2f}, "
                     f"{t['tokens'] / max(t['seconds'], 1e-9):,.0f} tokens/s here")
    return "\n".join(lines)


def pooling_config(stage, state):
    s, root = stage.cfg.train, stage.project.root
    mlm_dir = Path(stage.mlm_dir)
    initialized_from = (mlm_dir.relative_to(root) if mlm_dir.is_relative_to(root) else mlm_dir).as_posix()
    return {"pooling": "mean", "exclude_padding": True, "normalize": "l2", "similarity": "cosine",
            "query_prefix": stage.q_prefix, "passage_prefix": stage.p_prefix, "max_query_tokens": stage.max_q,
            "max_passage_tokens": stage.max_p,
            "note": "limits include the prefix and [CLS]/[SEP]; texts are normalized by the tokenizer only",
            "stage": "weakly supervised contrastive (pired.contrastive)",
            "trained_steps": state.step, "batch_size": s.batch_size, "temperature": s.temperature,
            "initialized_from": initialized_from}


def export_final_model(stage, model, state, val):
    final = Path(stage.cfg.final_dir)
    shutil.rmtree(final, ignore_errors=True)
    model.eval()
    model.save_pretrained(final)
    stage.tokenizer.backend_tokenizer.no_truncation()
    stage.tokenizer.save_pretrained(final)
    pooling = pooling_config(stage, state)
    final.joinpath("pooling_config.json").write_bytes(json.dumps(pooling, ensure_ascii=False, indent=1).encode("utf-8"))
    save_reference_outputs(model, final)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    write_text(f"# {stage.project.name} (stage 3: weakly supervised contrastive){' - SMOKE RUN' if stage.smoke else ''}"
               f"\n\nArabic encoder ({n_params:.0f}M parameters), trained from the MLM model of stage 2 on weak (query, "
               f"passage) pairs with InfoNCE. Embedding: mean over real tokens, L2-normalized. Queries start with "
               f"`{stage.q_prefix}`, passages with `{stage.p_prefix}`. The model code is `src/pired/encoder` "
               "of the project repo.\n", final / "README.md")
    reloaded = ArabicEncoderModel.from_pretrained(final).to(stage.device).eval()
    check = val.df["passage"].tolist()[:64]
    diff = (stage.make_embedder(model).encode(check, "passage")
            - stage.make_embedder(reloaded).encode(check, "passage")).abs().max().item()
    check_reference_outputs(reloaded, final, atol=0.0)
    if diff != 0.0 or json.loads((final / "config.json").read_text())["architectures"] != ["ArabicEncoderModel"]:
        raise AssertionError(f"the saved model differs from the trained one (max |diff| {diff})")
    log(f"saved to {final}: {sorted(p.name for p in final.iterdir())}; reloaded: identical embeddings, reference "
        "outputs reproduced, no MLM head")
    del reloaded
    push_folder(stage.api, final, stage.repo_id("model"), "model", stage.cfg.hub.push)
    return final


def eval_tasks(stage):
    e = stage.cfg.eval
    return get_tasks(e.tasks, e.languages, stage.smoke, e.smoke_queries, e.smoke_distractors, stage.token)


def our_mteb_encoder(stage, model, name):
    return ours_encoder(model, stage.tokenizer, name, (stage.q_prefix, stage.p_prefix), (stage.max_q, stage.max_p),
                        stage.cfg.eval.batch_tokens, stage.device, mask_full_chunks=False)


def evaluate_before(stage, val):
    s = stage.cfg.train
    results = load_eval_results(stage)
    before_val = evaluate_pairs(stage, stage.encoder, val, s.eval_batch_tokens, s.val_random_pairs)
    log(f"validation pairs ({len(val):,}, each query against all passages): {val_summary(before_val)}")
    log("  per source nDCG@10: " + ", ".join(f"{k} {v['ndcg@10']:.3f}" for k, v in before_val["per_source"].items()))
    mteb_scores = run_mteb_tasks(our_mteb_encoder(stage, stage.encoder, "local/mlm-encoder"), eval_tasks(stage))
    results["before (MLM encoder)"] = {"validation": before_val, **mteb_scores}
    save_eval_results(stage, results)
    free_gpu()
    return results


@torch.no_grad()
def evaluate_pairs_with(stage, encoder, val, n_random):
    q = torch.from_numpy(encoder.encode_texts(val.df["query"].tolist(), True)).to(stage.device)
    p = torch.from_numpy(encoder.encode_texts(val.df["passage"].tolist(), False)).to(stage.device)
    return pair_metrics(q, p, val, n_random, stage.device)


def evaluate_after(stage, val):
    cfg = stage.cfg
    results = load_eval_results(stage)
    free_gpu()
    final_model = ArabicEncoderModel.from_pretrained(cfg.final_dir).to(stage.device).eval()
    after_val = evaluate_pairs(stage, final_model, val, cfg.train.eval_batch_tokens, cfg.train.val_random_pairs)
    log(f"trained encoder, validation pairs: {val_summary(after_val)}")
    tasks = eval_tasks(stage)
    results["after (this stage)"] = {"validation": after_val, **run_mteb_tasks(
        our_mteb_encoder(stage, final_model, f"local/{cfg.train.run_name}"), tasks)}
    del final_model
    free_gpu()
    e5, e5_model = baseline_encoder(cfg.eval.baseline, cfg.eval.baseline_prefixes, cfg.eval.baseline_max_len,
                                    cfg.eval.batch_tokens, stage.device, mask_full_chunks=False)
    e5_val = evaluate_pairs_with(stage, e5, val, cfg.train.val_random_pairs)
    log(f"{cfg.eval.baseline}, validation pairs: {val_summary(e5_val)}")
    results[f"baseline ({cfg.eval.baseline.split('/')[-1]})"] = {"validation": e5_val, **run_mteb_tasks(e5, tasks)}
    del e5, e5_model
    free_gpu()
    save_eval_results(stage, results)
    log("\n" + md_table(results_header(stage), results_table(stage, results)))
    return results


def results_header(stage):
    return ["model", "val nDCG@10", "val R@10", "val R@100",
            *[f"{t.replace('Retrieval', '')} {m}" for t in stage.cfg.eval.tasks for m in ("nDCG@10", "R@100")]]


def results_table(stage, results):
    rows = []
    for name, r in results.items():
        row = [name, f"{r['validation']['ndcg@10']:.4f}", f"{r['validation']['recall@10']:.4f}",
               f"{r['validation']['recall@100']:.4f}"]
        for task in stage.cfg.eval.tasks:
            row += [f"{r[task]['ndcg@10']:.4f}", f"{r[task]['recall@100']:.4f}"] if task in r else ["-", "-"]
        rows.append(row)
    return rows
