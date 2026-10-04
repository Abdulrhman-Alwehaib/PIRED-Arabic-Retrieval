import contextlib
import dataclasses
import json
import shutil

import torch
from transformers import AutoTokenizer

from ..common.device import autocast_ctx, exact_fp32, free_gpu
from ..common.runlog import log
from ..encoder.weights import check_reference_outputs, load_encoder
from .training import (BatchLoader, TrainState, build_optimizer, build_scheduler, full_batch_step, grad_cache_step,
                       load_checkpoint, save_checkpoint, train_steps)

PADDING_TEXTS = ["ما هي عاصمة فرنسا؟",
                 "كتب المؤرخ عن تاريخ المدينة القديمة وأسواقها ومساجدها في القرن الثاني عشر الميلادي.",
                 "نعم",
                 "تقع المدينة على ضفاف النهر وتشتهر بصناعة النسيج والزراعة وتربية المواشي منذ مئات السنين."]
TOKENIZER_PARTS = ("normalizer", "pre_tokenizer", "model", "post_processor", "decoder", "added_tokens")


def stage3_model_test(stage):
    diffs = check_reference_outputs(stage.encoder, stage.stage3_dir, atol=1e-4)
    return ("PASS strict=True load; outputs on stage 3's fixed input reproduced: max |diff| "
            + ", ".join(f"{k} {v:.1e}" for k, v in diffs.items()))


def tokenizer_test(stage):
    t3 = json.loads((stage.stage3_dir / "tokenizer.json").read_text(encoding="utf-8"))
    t1_path = stage.project.path("tokenizer_dir") / "tokenizer.json"
    t1 = json.loads(t1_path.read_text(encoding="utf-8")) if t1_path.exists() else t3
    ours = json.loads(stage.tokenizer.backend_tokenizer.to_str())
    same = {k: ours[k] == t3[k] == t1[k] for k in TOKENIZER_PARTS}
    samples = ["أَهْلاً وسَهْلاً ١٢٣", "ما هي عاصمة قطر؟", "Apple iPhone 15 بسعر 999$"]
    stage3_tok = AutoTokenizer.from_pretrained(stage.stage3_dir)
    same_ids = all(stage.tokenizer(t)["input_ids"] == stage3_tok(t)["input_ids"] for t in samples)
    if not (all(same.values()) and same_ids):
        raise AssertionError(f"the tokenizer differs from stage 3: {same}")
    return (f"PASS normalizer, pre-tokenizer, vocabulary/merges, post-processor, decoder and special tokens equal to "
            f"the stage-3 tokenizer{' and the stage-1 file' if t1 is not t3 else ''}; fingerprint "
            f"{stage.tokenizer_fingerprint}; same ids on {len(samples)} samples; prefixes {stage.q_prefix!r} / "
            f"{stage.p_prefix!r} and limits {stage.max_q}/{stage.max_p} = stage 3's pooling_config.json")


def pooling_test(stage, val):
    encoder, device = stage.encoder, stage.device
    ids = stage.tokenize_texts(PADDING_TEXTS, "passage")

    def padding_diff(use_autocast):
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

    d32, d16 = padding_diff(False), padding_diff(True)
    e = stage.embedder.encode(val.q["query"].tolist()[:500], "query")
    norm_err = (e.norm(dim=-1) - 1).abs().max().item()
    if not (d32 < 1e-5 and d16 < 2e-2 and norm_err < 1e-5):
        raise AssertionError(f"pooling test failed: {d32}, {d16}, {norm_err}")
    return (f"PASS 4 texts of {sorted(map(len, ids))} tokens: alone vs padded batch (and random pad content) max |diff| "
            f"{d32:.1e} in fp32, {d16:.1e} with bf16 autocast; {len(e)} embeddings, max | ||e|| - 1 | = {norm_err:.1e}")


def grad_cache_test(stage, train_data, n_neg):
    encoder, device, s = stage.encoder, stage.device, stage.cfg.train
    loader = BatchLoader(stage, train_data, n_neg, 4, seed=0, chunk_tokens=s.chunk_tokens)
    source = loader.schedule[0][0]
    small = loader.rows_batch(source, loader.perms[source][:4], chunk_tokens=300)

    def grads(step_fn, use_autocast):
        encoder.train()
        encoder.zero_grad(set_to_none=True)
        loss, _ = step_fn(encoder, small, s, device, use_autocast)
        return loss.item(), torch.cat([p.grad.flatten().float() for p in encoder.parameters()])

    res = {}
    for use_autocast in (False, True):
        with contextlib.nullcontext() if use_autocast else exact_fp32():
            l_full, g_full = grads(full_batch_step, use_autocast)
            l_gc, g_gc = grads(grad_cache_step, use_autocast)
        res[use_autocast] = ((g_gc - g_full).norm() / g_full.norm()).item(), abs(l_gc - l_full)
    encoder.zero_grad(set_to_none=True)
    encoder.eval()
    if res[False][0] >= 1e-4:
        raise AssertionError(f"GradCache gradients differ from the full-batch ones: {res}")
    return (f"PASS 4 queries + {4 * (1 + n_neg)} passages in {len(small.q_chunks)} + {len(small.p_chunks)} chunks: "
            f"relative gradient difference {res[False][0]:.1e} in exact fp32 (loss difference {res[False][1]:.1e}), "
            f"{res[True][0]:.1e} with bf16 autocast")


def overfit_test(stage, train_data, n_neg):
    device, s = stage.device, stage.cfg.train
    loader = BatchLoader(stage, train_data, n_neg, 4, seed=0, chunk_tokens=s.chunk_tokens)
    source = loader.schedule[0][0]
    model = load_encoder(stage.stage3_dir).to(device).train()
    optimizer = build_optimizer(model, dataclasses.replace(s, lr=5e-5), device)
    fixed = loader.rows_batch(source, loader.perms[source][:8], s.chunk_tokens)
    curve = {}
    for k in range(61):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = grad_cache_step(model, fixed, s, device)
        if k % 10 == 0:
            curve[k] = round(loss.item(), 4)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    del model, optimizer
    free_gpu()
    if curve[60] >= 0.05:
        raise AssertionError(f"the loss on a fixed batch did not go to zero: {curve}")
    return f"PASS 8 fixed queries x {1 + n_neg} passages, AdamW 5e-5, loss by step {curve}"


def resume_test(stage, train_data, n_neg):
    device, s = stage.device, stage.cfg.train

    def run(stop_at, resume_from, seed):
        torch.manual_seed(seed)
        model = load_encoder(stage.stage3_dir).to(device)
        loader = BatchLoader(stage, train_data, n_neg, 4, seed=7, chunk_tokens=s.chunk_tokens)
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
    b2, _, objs = run(None, ckpt, 123)
    del objs
    shutil.rmtree(stage.work / "test_resume", ignore_errors=True)
    free_gpu()
    diff = max(abs(x - y) / max(abs(x), 1e-6) for x, y in zip(a, b1 + b2))
    if not (len(b1 + b2) == len(a) == 6 and diff < 2e-2):
        raise AssertionError(f"resumed run differs: {a} vs {b1} + {b2}")
    return (f"PASS 3 steps + checkpoint + 3 resumed steps (other seed) vs 6 straight steps: max relative loss "
            f"difference {diff:.1e}; weights, optimizer, lr schedule and data position restored")


def run_selftests(stage, train_data, val, n_neg, stats1, stats2):
    results = {"1 stage-3 model": stage3_model_test(stage), "2 tokenizer": tokenizer_test(stage),
               "3 pooling": pooling_test(stage, val),
               "4 negatives": stats2.get("test4", "not found in Part 2's stats"),
               "5 decontamination": stats1.get("decontam_test", "not found in Part 1's stats")}
    if not (results["4 negatives"].startswith("PASS") and results["5 decontamination"].startswith("PASS")):
        raise AssertionError(f"Parts 1-2 tests did not pass: {results}")
    results["6 GradCache"] = grad_cache_test(stage, train_data, n_neg)
    results["7 overfit"] = overfit_test(stage, train_data, n_neg)
    results["8 resume"] = resume_test(stage, train_data, n_neg)
    log("\n".join(f"{k:<18} {v}" for k, v in results.items()))
    return results
