import dataclasses
import itertools
import time

import torch

from ..common.runlog import log
from ..encoder.modeling import ArabicEncoderForMaskedLM
from .checkpoints import TrainState
from .collator import MLMCollator
from .data import PackedTextData
from .schedule import build_optimizer, build_scheduler
from .trainer import make_model_config, train_steps


def stream_sample_texts(repo_id, shard, n_docs):
    from datasets import load_dataset

    rows = load_dataset("parquet", data_files=[f"hf://datasets/{repo_id}/{shard}"], split="train", streaming=True)
    return [row["text"] for row in itertools.islice(rows, n_docs)]


def sample_token_ids(cfg, tokenizer):
    texts = stream_sample_texts(cfg.benchmark.dataset_repo, cfg.benchmark.shard, cfg.benchmark.n_docs)
    docs_ids = tokenizer(texts, add_special_tokens=False, verbose=False)["input_ids"]
    log(f"tokenizer vocab {len(tokenizer):,}; sample: {len(texts):,} docs, {sum(map(len, docs_ids)):,} tokens")
    return docs_ids


def measure_throughput(cfg, tokenizer, docs_ids, micro_batch_size, warmup, timed, compile_model, device, logger=None):
    s = cfg.train
    data = PackedTextData(docs_ids, s.seq_len, micro_batch_size, tokenizer.cls_token_id, tokenizer.sep_token_id)
    torch.manual_seed(0)
    model = ArabicEncoderForMaskedLM(make_model_config(cfg.model, tokenizer)).to(device)
    if compile_model:
        model.model.compile()
    run = dataclasses.replace(s, micro_batch_size=micro_batch_size, grad_accum_steps=1,
                              warmup_steps=min(50, warmup + timed), total_steps=warmup + timed)
    optimizer = build_optimizer(model, run, device)
    scheduler = build_scheduler(optimizer, run)
    collator = MLMCollator.from_tokenizer(tokenizer, cfg.mlm, seed=0)
    state = TrainState()
    kw = dict(grad_accum_steps=1, max_grad_norm=run.max_grad_norm, device=device)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    first = train_steps(model, optimizer, scheduler, data, collator, state, num_steps=warmup, record_losses=True, **kw)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    train_steps(model, optimizer, scheduler, data, collator, state, num_steps=timed, logger=logger,
                log_every=max(1, timed // 6), **kw)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t1
    result = {"micro_batch_size": micro_batch_size, "tokens_per_sec": timed * micro_batch_size * s.seq_len / seconds,
              "first_loss": first[0], "warmup_seconds": warmup_seconds,
              "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return result


def measure_with_oom_backoff(cfg, tokenizer, docs_ids, micro_batch_size, **kw):
    while True:
        try:
            return measure_throughput(cfg, tokenizer, docs_ids, micro_batch_size, **kw)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if micro_batch_size == 1:
                raise
            micro_batch_size //= 2
            log(f"out of memory -> retrying with micro_batch_size={micro_batch_size}")


def recommend_settings(cfg, result, last_record):
    s = cfg.train
    tokens_per_step = s.micro_batch_size * s.grad_accum_steps * s.seq_len
    tps = result["tokens_per_sec"]
    plan_tokens = s.total_steps * tokens_per_step
    fit, per_step = result["micro_batch_size"], s.micro_batch_size * s.grad_accum_steps
    return "\n".join([
        f"GPU: {torch.cuda.get_device_name(0)}  micro_batch={fit} x {s.seq_len}",
        f"loss: {result['first_loss']:.3f} (step 1) -> {last_record['loss']:.3f} (step {last_record['step']})",
        f"throughput: {tps:,.0f} tokens/sec   peak memory: {result['peak_mem_gb']:.1f} GiB",
        f"time for total_steps={s.total_steps:,} ({plan_tokens / 1e9:.1f}B tokens) on this GPU: "
        f"{plan_tokens / tps / 3600:.1f} h ({plan_tokens / tps / 86400:.1f} days)",
        f"with {tokens_per_step:,} tokens/step: {tokens_per_step / tps:.2f} s/step -> save_every_steps ~ "
        f"{max(1, round(1800 * tps / tokens_per_step))} for a checkpoint every ~30 min",
        f"-> micro_batch_size = {fit}, grad_accum_steps = {per_step // fit} ({fit} x {per_step // fit} = {per_step} "
        "sequences per step)",
    ])


def compile_benchmark(cfg, tokenizer, docs_ids, device):
    micro = cfg.benchmark.micro_batch_size
    bench = {}
    for compiled in (False, True):
        try:
            r = measure_with_oom_backoff(cfg, tokenizer, docs_ids, micro, warmup=10,
                                         timed=cfg.benchmark.compile_bench_steps, compile_model=compiled,
                                         device=device)
            micro = r["micro_batch_size"]
            bench[compiled] = r
            log(f"compile={compiled}: {r['tokens_per_sec']:,.0f} tokens/sec, peak {r['peak_mem_gb']:.1f} GiB, first "
                f"10 steps {r['warmup_seconds']:.0f}s (includes compilation)")
        except Exception as e:
            log(f"compile={compiled} failed: {type(e).__name__}: {str(e)[:300]}")
    if len(bench) == 2:
        log(f"speedup from torch.compile: {bench[True]['tokens_per_sec'] / bench[False]['tokens_per_sec']:.2f}x")
    return bench
