import contextlib
import dataclasses
import math
import shutil

import torch
from safetensors.torch import load_file

from ..common.device import autocast_ctx, exact_fp32, free_gpu
from ..common.io import write_json
from ..common.runlog import log
from ..encoder.modeling import ArabicEncoderForMaskedLM
from ..encoder.weights import load_encoder_without_head
from .run import run_dir
from .training import (BatchLoader, TrainState, batch_loss, build_optimizer, build_scheduler, embed_chunk,
                       full_batch_step, grad_cache_step, load_checkpoint, make_batch, save_checkpoint, train_steps)

PADDING_TEXTS = ["ما هي عاصمة فرنسا؟",
                 "كتب المؤرخ عن تاريخ المدينة القديمة وأسواقها ومساجدها في القرن الثاني عشر الميلادي.",
                 "نعم",
                 "تقع المدينة على ضفاف النهر وتشتهر بصناعة النسيج والزراعة وتربية المواشي منذ مئات السنين."]


def encoder_load_test(stage, skipped_keys):
    encoder = stage.encoder
    with torch.device("meta"):
        head = {k for k in ArabicEncoderForMaskedLM(encoder.config).state_dict() if k.startswith("head.")}
    stored = set(load_file(str(stage.mlm_dir / "model.safetensors")).keys())
    if set(skipped_keys) != stored & head or not all(k.startswith("head.") for k in skipped_keys):
        raise AssertionError(f"skipped keys {skipped_keys} are not exactly the stored MLM head keys")
    if {f"model.{k}" for k in encoder.state_dict()} != stored - set(skipped_keys):
        raise AssertionError("the encoder tensors differ from the checkpoint's model.* tensors")
    return f"PASS strict=True; {len(encoder.state_dict())} encoder tensors loaded; skipped only the MLM head: {skipped_keys}"


def padding_diff(stage, ids, use_autocast):
    encoder, device = stage.encoder, stage.device
    encoder.eval()
    width = max(map(len, ids))
    batch = torch.full((len(ids), width), stage.pad_id, dtype=torch.long)
    mask = torch.zeros_like(batch)
    for i, x in enumerate(ids):
        batch[i, : len(x)], mask[i, : len(x)] = torch.tensor(x), 1
    garbage = batch.clone()
    garbage[mask == 0] = torch.randint(10, len(stage.tokenizer), (int((mask == 0).sum()),))
    ctx = autocast_ctx(device) if use_autocast else exact_fp32()
    with torch.no_grad(), ctx:
        batched = encoder.embed(batch.to(device), mask.to(device))
        batched_garbage = encoder.embed(garbage.to(device), mask.to(device))
        alone = torch.cat([encoder.embed(torch.tensor([x], device=device)) for x in ids])
    return max((batched - alone).abs().max().item(), (batched_garbage - batched).abs().max().item())


def padding_test(stage):
    ids = stage.embedder.tokenize(PADDING_TEXTS, "passage")
    d32, d16 = padding_diff(stage, ids, False), padding_diff(stage, ids, True)
    if not (d32 < 1e-5 and d16 < 2e-2):
        raise AssertionError(f"padding changes the embedding: fp32 {d32}, bf16 {d16}")
    return (f"PASS 4 texts of {sorted(map(len, ids))} tokens: alone vs padded batch (and random pad content) max |diff| "
            f"{d32:.1e} in fp32, {d16:.1e} with bf16 autocast")


def unit_length_test(stage, val):
    e = stage.embedder.encode(val.df["query"].tolist()[:500], "query")
    err = (e.norm(dim=-1) - 1).abs().max().item()
    if err >= 1e-5:
        raise AssertionError(f"embeddings are not unit length: {err}")
    return f"PASS {len(e)} embeddings, max | ||e|| - 1 | = {err:.1e}"


def initial_loss_test(stage, test_batch):
    encoder, device, s = stage.encoder, stage.device, stage.cfg.train
    encoder.eval()
    with torch.no_grad():
        reps = {side: torch.cat([embed_chunk(encoder, i, m, True, device).float() for _, i, m in chunks])[
            torch.argsort(torch.cat([r for r, _, _ in chunks]).to(device))]
            for side, chunks in (("q", test_batch.q_chunks), ("p", test_batch.p_chunks))}
        loss0, st0 = batch_loss(reps["q"], reps["p"], test_batch, s, device)
    ln_b = math.log(test_batch.size)
    if loss0.item() > ln_b:
        raise AssertionError(f"initial loss {loss0.item()} > ln(batch) {ln_b}")
    return (f"PASS {loss0.item():.3f} <= ln({test_batch.size}) = {ln_b:.3f} (query->passage {st0['loss_qp']:.3f}, "
            f"passage->query {st0['loss_pq']:.3f}; in-batch accuracy {st0['acc_qp']:.2f}; temperature {s.temperature})")


def grad_cache_test(stage, train_data, test_loader, test_batch):
    encoder, device, s = stage.encoder, stage.device, stage.cfg.train
    small = make_batch(stage, *train_data.rows(test_batch.source, test_loader.perms[test_batch.source][:16]),
                       test_batch.source, chunk_tokens=200)

    def grads(step_fn, use_autocast):
        encoder.train()
        encoder.zero_grad(set_to_none=True)
        loss, _ = step_fn(encoder, small, s, device, use_autocast)
        return loss.item(), torch.cat([p.grad.flatten().float() for p in encoder.parameters()])

    results = {}
    for use_autocast in (False, True):
        with contextlib.nullcontext() if use_autocast else exact_fp32():
            l_full, g_full = grads(full_batch_step, use_autocast)
            l_gc, g_gc = grads(grad_cache_step, use_autocast)
        results[use_autocast] = ((g_gc - g_full).norm() / g_full.norm()).item(), abs(l_gc - l_full)
    encoder.zero_grad(set_to_none=True)
    encoder.eval()
    if results[False][0] >= 1e-4:
        raise AssertionError(f"GradCache gradients differ from the full-batch ones: {results}")
    return (f"PASS 16 pairs in {len(small.q_chunks)} query + {len(small.p_chunks)} passage chunks: relative gradient "
            f"difference {results[False][0]:.1e} in exact fp32 (loss difference {results[False][1]:.1e}), "
            f"{results[True][0]:.1e} with bf16 autocast")


def overfit_test(stage, train_data, test_loader, test_batch):
    device, s = stage.device, stage.cfg.train
    model = load_encoder_without_head(stage.mlm_dir)[0].to(device).train()
    optimizer = build_optimizer(model, dataclasses.replace(s, lr=5e-5), device)
    fixed = make_batch(stage, *train_data.rows(test_batch.source, test_loader.perms[test_batch.source][:32]),
                       test_batch.source, s.chunk_tokens)
    curve = {}
    for k in range(81):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = grad_cache_step(model, fixed, s, device)
        if k % 10 == 0:
            curve[k] = round(loss.item(), 4)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    del model, optimizer
    free_gpu()
    if curve[80] >= 0.05:
        raise AssertionError(f"the loss on a fixed batch did not go to zero: {curve}")
    return f"PASS 32 fixed pairs, AdamW 5e-5, loss by step {curve}"


def resume_test(stage, train_data):
    device, s = stage.device, stage.cfg.train

    def run(stop_at, resume_from, seed):
        torch.manual_seed(seed)
        model = load_encoder_without_head(stage.mlm_dir)[0].to(device)
        loader = BatchLoader(stage, train_data, 16, seed=7, chunk_tokens=s.chunk_tokens)
        optimizer = build_optimizer(model, s, device)
        scheduler = build_scheduler(optimizer, 6, s)
        state = load_checkpoint(resume_from, model, optimizer, scheduler, loader) if resume_from else TrainState()
        losses = train_steps(stage, model, optimizer, scheduler, loader, state, total_steps=6, stop_at=stop_at,
                             record_losses=True)
        return losses, state, (model, optimizer, scheduler, loader)

    a, _, objs = run(None, None, 0)
    del objs
    free_gpu()
    b1, state_b, objs = run(3, None, 0)
    ckpt = save_checkpoint(stage.work / "test_resume", *objs[:3], state_b, objs[3], s)
    del objs
    free_gpu()
    b2, state_c, objs = run(None, ckpt, 123)
    del objs
    shutil.rmtree(stage.work / "test_resume", ignore_errors=True)
    free_gpu()
    diff = max(abs(x - y) / abs(x) for x, y in zip(a, b1 + b2))
    if not (len(b1 + b2) == len(a) == 6 and diff < 2e-2):
        raise AssertionError(f"resumed run differs: {a} vs {b1} + {b2}")
    return (f"PASS 3 steps + checkpoint + 3 resumed steps (other seed) vs 6 straight steps: max relative loss "
            f"difference {diff:.1e} (allowed 2e-2, bf16 kernels need not be bit-deterministic); ended at step "
            f"{state_c.step} with weights, optimizer, lr schedule and data position restored")


def run_selftests(stage, skipped_keys, train_data, val):
    results = {}
    test_loader = BatchLoader(stage, train_data, 64, seed=0, chunk_tokens=stage.cfg.train.chunk_tokens)
    test_batch = test_loader.get(0)
    log(f"train pairs per source: {train_data.sizes}; validation pairs: {len(val):,}; test batch: 64 pairs from "
        f"'{test_batch.source}'")
    tests = [("1 encoder load", lambda: encoder_load_test(stage, skipped_keys)),
             ("2 padding", lambda: padding_test(stage)),
             ("3 unit length", lambda: unit_length_test(stage, val)),
             ("4 initial loss", lambda: initial_loss_test(stage, test_batch)),
             ("5 GradCache", lambda: grad_cache_test(stage, train_data, test_loader, test_batch)),
             ("6 overfit", lambda: overfit_test(stage, train_data, test_loader, test_batch)),
             ("7 resume", lambda: resume_test(stage, train_data))]
    for name, test in tests:
        results[name] = test()
        log(f"{name:<16} {results[name]}")
    write_json(results, run_dir(stage) / "test_results.json")
    return results
