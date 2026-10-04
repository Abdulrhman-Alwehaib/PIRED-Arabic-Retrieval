import dataclasses
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from pired.common.runlog import JsonlLogger, read_jsonl
from pired.encoder.modeling import ArabicEncoderForMaskedLM
from pired.pretraining.checkpoints import TrainState, load_checkpoint, save_checkpoint
from pired.pretraining.collator import MLMCollator
from pired.pretraining.config import DataSettings, MLMSettings, TrainSettings
from pired.pretraining.corpus import pack
from pired.pretraining.data import PretrainingData, SyntheticData
from pired.pretraining.schedule import build_optimizer, build_scheduler, decay_start_step, lr_factor
from pired.pretraining.trainer import parameter_breakdown, train_steps
from pired.tokenizer.fingerprint import vocab_fingerprint
from conftest import pad_batch, random_sequences, tiny_config

DEVICE = torch.device("cpu")
SPECIALS = [0, 1, 2, 3, 4]


def tiny_collator(seed=0):
    return MLMCollator(1000, 0, 4, SPECIALS, MLMSettings(), seed=seed)


def test_collator_statistics():
    collator = tiny_collator(seed=7)
    seqs = random_sequences([128] * 48 + [70] * 16, seed=7)
    original, _ = pad_batch(seqs)
    b = collator(seqs)
    selected = b["labels"] != -100
    maskable = ~torch.isin(original, torch.tensor(SPECIALS))
    assert not (selected & ~maskable).any()
    assert torch.equal(b["labels"][selected], original[selected])
    assert torch.equal(b["input_ids"][~selected], original[~selected])
    frac_selected = selected.sum().item() / maskable.sum().item()
    became_mask = (b["input_ids"][selected] == 4).float().mean().item()
    unchanged = (b["input_ids"][selected] == original[selected]).float().mean().item()
    assert abs(frac_selected - 0.30) < 0.01 and abs(became_mask - 0.8) < 0.02 and abs(unchanged - 0.1) < 0.02
    assert "attention_mask" in b and "attention_mask" not in collator(random_sequences([64, 64]))


def test_parameter_breakdown_of_the_real_size():
    from pired.encoder.configuration import ArabicEncoderConfig

    with torch.device("meta"):
        model = ArabicEncoderForMaskedLM(ArabicEncoderConfig())
    parts = parameter_breakdown(model)
    assert round(parts["encoder only (what stages 3-5 use)"] / 1e6) == 134


def build_tiny_run(seed, settings):
    torch.manual_seed(seed)
    model = ArabicEncoderForMaskedLM(tiny_config())
    optimizer = build_optimizer(model, settings, DEVICE)
    scheduler = build_scheduler(optimizer, settings)
    data = SyntheticData(1000, 5, 2, 3, settings.seq_len, settings.micro_batch_size, seed=seed)
    return model, optimizer, scheduler, data, tiny_collator(seed=seed)


DRY = TrainSettings(run_name="dry-run", seq_len=64, micro_batch_size=4, grad_accum_steps=2, total_steps=20, lr=2e-3,
                    warmup_steps=4, log_every=5)
COMMON = dict(grad_accum_steps=DRY.grad_accum_steps, max_grad_norm=DRY.max_grad_norm, device=DEVICE)


def test_resume_reproduces_an_uninterrupted_run():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        losses_a = train_steps(*build_tiny_run(0, DRY), TrainState(), num_steps=20, record_losses=True, **COMMON)
        model, optimizer, scheduler, data, collator = build_tiny_run(0, DRY)
        state = TrainState()
        logger = JsonlLogger(tmp / "log.jsonl", echo=False)
        saved = []
        losses_b = train_steps(model, optimizer, scheduler, data, collator, state, num_steps=10, record_losses=True,
                               logger=logger, log_every=DRY.log_every, save_every=10,
                               on_checkpoint=lambda st: saved.append(save_checkpoint(
                                   tmp / "ckpt", model, optimizer, scheduler, data, collator, st, DRY)), **COMMON)
        ckpt = saved[-1]
        assert sorted(p.name for p in ckpt.iterdir()) == ["config.json", "model.safetensors", "training_state.pt"]
        model, optimizer, scheduler, data, collator = build_tiny_run(123, DRY)
        state = load_checkpoint(ckpt, model, optimizer, scheduler, data, collator, DRY)
        assert state.step == 10 and data.position == 20
        losses_b += train_steps(model, optimizer, scheduler, data, collator, state, num_steps=10, record_losses=True,
                                logger=logger, log_every=DRY.log_every, **COMMON)
        records = read_jsonl(tmp / "log.jsonl")
    assert max(abs(a - b) / abs(a) for a, b in zip(losses_a, losses_b)) < 1e-3
    assert [r["step"] for r in records] == [5, 10, 15, 20] and {"loss", "lr", "tokens_per_sec"} <= set(records[0])


def test_wsd_schedule_and_pre_decay_continuation():
    wsd_long = dataclasses.replace(DRY, run_name="wsd-long", lr_schedule="wsd", decay_fraction=0.25, total_steps=30)
    wsd_short = dataclasses.replace(wsd_long, run_name="wsd-short", total_steps=20)
    short_start, long_start = decay_start_step(wsd_short), decay_start_step(wsd_long)
    f_short = [lr_factor(k, wsd_short) for k in range(wsd_short.total_steps + 1)]
    f_long = [lr_factor(k, wsd_long) for k in range(wsd_long.total_steps + 1)]
    assert (short_start, long_start) == (15, 22)
    assert f_short[: short_start + 1] == f_long[: short_start + 1]
    assert all(f == 1.0 for f in f_long[wsd_long.warmup_steps - 1: long_start + 1])
    assert all(a > b for a, b in zip(f_short[short_start:], f_short[short_start + 1:]))
    assert math.isclose(f_short[-1], wsd_short.min_lr_ratio)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        losses_a = train_steps(*build_tiny_run(0, wsd_long), TrainState(), num_steps=wsd_long.total_steps,
                               record_losses=True, **COMMON)
        model, optimizer, scheduler, data, collator = build_tiny_run(0, wsd_short)
        pre_decay = []
        train_steps(model, optimizer, scheduler, data, collator, TrainState(), num_steps=wsd_short.total_steps,
                    pre_decay_step=short_start,
                    on_pre_decay=lambda st: pre_decay.append(save_checkpoint(
                        tmp / "pre-decay", model, optimizer, scheduler, data, collator, st, wsd_short)), **COMMON)
        assert [p.name for p in pre_decay] == [f"step-{short_start:08d}"]
        assert math.isclose(scheduler.get_last_lr()[0], wsd_short.lr * wsd_short.min_lr_ratio)
        model, optimizer, scheduler, data, collator = build_tiny_run(123, wsd_long)
        state = load_checkpoint(pre_decay[0], model, optimizer, scheduler, data, collator)
        assert state.step == short_start and math.isclose(scheduler.get_last_lr()[0], wsd_long.lr)
        losses_b = train_steps(model, optimizer, scheduler, data, collator, state,
                               num_steps=wsd_long.total_steps - state.step, record_losses=True, **COMMON)
        assert math.isclose(scheduler.get_last_lr()[0], wsd_long.lr * wsd_long.min_lr_ratio)
    assert len(losses_b) == wsd_long.total_steps - short_start
    assert max(abs(a - b) / abs(a) for a, b in zip(losses_a[short_start:], losses_b)) < 1e-3


class FakeTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab = {f"t{i}": i for i in range(vocab_size)}

    def get_vocab(self):
        return self.vocab


def write_fake_shards(root, sizes, seq_len, vocab):
    (root / "shards").mkdir(parents=True)
    shards = []
    for s, n in enumerate(sizes):
        rows = np.zeros((n, seq_len), dtype=np.uint16)
        rows[:, 0], rows[:, -1] = 2, 3
        rows[:, 1], rows[:, 2] = 100 + s, 200 + np.arange(n)
        np.save(root / "shards" / f"s{s}.npy", rows)
        shards.append({"file": f"shards/s{s}.npy", "source": "fake", "n_sequences": n})
    np.save(root / "validation.npy", np.full((3, seq_len), 7, dtype=np.uint16))
    manifest = {"seq_len": seq_len, "vocab_fingerprint": vocab_fingerprint(vocab), "shards": shards,
                "validation": {"file": "validation.npy", "n_sequences": 3}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_pretraining_data_loader():
    sizes = [5, 7, 3, 9, 4]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_fake_shards(root, sizes, 16, FakeTokenizer().get_vocab())
        settings = DataSettings(local_dir=root, window_shards=2)

        def make_loader(seed=0):
            return PretrainingData(FakeTokenizer(), 16, 4, seed, settings)

        loader = make_loader()
        epoch1 = np.concatenate([next(loader) for _ in range(7)])
        epoch2 = np.concatenate([next(loader) for _ in range(7)])
        every_row = sorted((100 + s, 200 + r) for s, n in enumerate(sizes) for r in range(n))
        for epoch in (epoch1, epoch2):
            assert sorted(zip(epoch[:, 1].tolist(), epoch[:, 2].tolist())) == every_row
        assert not np.array_equal(epoch1, epoch2)
        first = make_loader()
        for _ in range(3):
            next(first)
        saved = first.state_dict()
        expected = [next(first) for _ in range(9)]
        resumed = make_loader(seed=123)
        resumed.load_state_dict(saved)
        assert all(np.array_equal(x, y) for x, y in zip(expected, (next(resumed) for _ in range(9))))
        with pytest.raises(ValueError):
            PretrainingData(FakeTokenizer(999), 16, 4, 0, settings)
        assert loader.validation_sequences(2).shape == (2, 16)
        for opened in (loader, first, resumed):
            opened.close()
    packed = tiny_collator()(epoch1[:4])
    assert set(packed) == {"input_ids", "labels"} and packed["input_ids"].dtype == torch.long
    assert (packed["labels"][:, [0, -1]] == -100).all()


def test_pack_rows():
    stream = [np.arange(10, 30, dtype=np.uint16), np.array([3], dtype=np.uint16)]
    rows = pack(stream, 8, 2, 3)
    assert rows.shape == (3, 8) and (rows[:, 0] == 2).all() and (rows[:, -1] == 3).all()
    assert rows[0, 1:-1].tolist() == list(range(10, 16))
    assert pack([], 8, 2, 3).shape == (0, 8)
