import importlib.util
import json
import sys
import math
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from pired.common.device import exact_fp32
from pired.encoder.modeling import ArabicEncoderForMaskedLM, ArabicEncoderModel
from pired.encoder.weights import (check_reference_outputs, load_weights_strict, reference_forward,
                                            reference_inputs, save_reference_outputs)
from pired.pretraining.collator import MLMCollator
from pired.pretraining.config import MLMSettings
from conftest import ROOT, pad_batch, random_sequences, tiny_config


def tiny_collator(seed=0):
    return MLMCollator(1000, 0, 4, [0, 1, 2, 3, 4], MLMSettings(), seed=seed)


def assert_tied(model):
    dec, emb = model.head.decoder.weight, model.model.embeddings.tok_embeddings.weight
    assert dec is emb and dec.data_ptr() == emb.data_ptr()


def test_shapes():
    torch.manual_seed(0)
    tiny = ArabicEncoderForMaskedLM(tiny_config()).eval()
    seqs = random_sequences([17, 9, 12])
    ids, mask = pad_batch(seqs)
    with torch.no_grad():
        enc_out = tiny.model(ids, mask, output_hidden_states=True)
        full_logits = tiny(ids, mask).logits
        batch = tiny_collator()(seqs)
        mlm_out = tiny(**batch)
        emb = tiny.model.embed(ids, mask)
    n_masked = int((batch["labels"] != -100).sum())
    assert enc_out.last_hidden_state.shape == (3, 17, 64)
    assert len(enc_out.hidden_states) == 3 and all(h.shape == (3, 17, 64) for h in enc_out.hidden_states)
    assert full_logits.shape == (3, 17, 1000)
    assert mlm_out.logits.shape == (n_masked, 1000) and mlm_out.loss.ndim == 0
    assert emb.shape == (3, 64) and torch.allclose(emb.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_padding_invariance():
    torch.manual_seed(0)
    model = ArabicEncoderForMaskedLM(tiny_config()).eval()
    seqs = random_sequences([12, 7, 20, 3], seed=1)
    ids, mask = pad_batch(seqs)
    garbage = ids.clone()
    garbage[mask == 0] = torch.randint(5, 1000, (int((mask == 0).sum()),))
    with torch.no_grad(), exact_fp32():
        batched = model.model(ids, mask).last_hidden_state
        batched_garbage = model.model(garbage, mask).last_hidden_state
        batched_emb = model.model.embed(ids, mask)
        for i, s in enumerate(seqs):
            single = torch.tensor([s])
            assert (batched[i, : len(s)] - model.model(single).last_hidden_state[0]).abs().max() < 1e-5
            assert (batched_emb[i] - model.model.embed(single)[0]).abs().max() < 1e-5
            assert (model(ids, mask).logits[i, : len(s)] - model(single).logits[0]).abs().max() < 1e-5
        assert (batched_garbage - batched)[mask.bool()].abs().max() < 1e-5


def test_weight_tying():
    torch.manual_seed(0)
    model = ArabicEncoderForMaskedLM(tiny_config())
    assert_tied(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model(**tiny_collator()(random_sequences([16, 16]))).loss.backward()
    opt.step()
    assert_tied(model)


@torch.no_grad()
def all_outputs(model, ids, mask):
    model.eval()
    with exact_fp32():
        return {"hidden": model.model(ids, mask).last_hidden_state, "embed": model.model.embed(ids, mask),
                "logits": model(ids, mask).logits}


def safetensors_keys(directory):
    keys = []
    for f in sorted(Path(directory).glob("model*.safetensors")):
        with safe_open(str(f), framework="pt") as st:
            keys.extend(st.keys())
    return sorted(keys)


def test_save_and_load():
    torch.manual_seed(0)
    src = ArabicEncoderForMaskedLM(tiny_config())
    with torch.no_grad():
        for p in src.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ids, mask = pad_batch(random_sequences([20, 11, 5], seed=2))
    expected = all_outputs(src, ids, mask)
    with tempfile.TemporaryDirectory() as tmp:
        mlm_dir, enc_dir = Path(tmp) / "mlm", Path(tmp) / "encoder"
        src.save_pretrained(mlm_dir)
        src.save_encoder(enc_dir)
        save_reference_outputs(src, mlm_dir)
        loaded, info = ArabicEncoderForMaskedLM.from_pretrained(mlm_dir, output_loading_info=True)
        assert not {k: v for k, v in info.items() if v}
        assert_tied(loaded)
        got = all_outputs(loaded, ids, mask)
        assert all(torch.equal(expected[k], got[k]) for k in expected)
        strict = load_weights_strict(ArabicEncoderForMaskedLM(tiny_config()), mlm_dir)
        assert_tied(strict)
        assert all(torch.equal(expected[k], all_outputs(strict, ids, mask)[k]) for k in expected)
        check_reference_outputs(strict, mlm_dir, atol=0.0)
        enc_keys = safetensors_keys(enc_dir)
        assert not any(k.startswith(("head.", "model.")) for k in enc_keys)
        enc_loaded = load_weights_strict(ArabicEncoderModel(tiny_config()), enc_dir).eval()
        with torch.no_grad(), exact_fp32():
            assert torch.equal(enc_loaded(ids, mask).last_hidden_state, expected["hidden"])
            assert torch.equal(enc_loaded.embed(ids, mask), expected["embed"])
        assert json.loads((enc_dir / "config.json").read_text())["architectures"] == ["ArabicEncoderModel"]


@pytest.mark.parametrize("vocab,lengths", [(1000, [64] * 8), (64_000, [256] * 4)])
def test_initial_loss_near_ln_vocab(vocab, lengths):
    torch.manual_seed(0)
    config = tiny_config(vocab_size=vocab) if vocab == 1000 else tiny_config(vocab_size=vocab, hidden_size=768,
                                                                             num_hidden_layers=12, num_attention_heads=12,
                                                                             intermediate_size=2048,
                                                                             max_position_embeddings=1024)
    model = ArabicEncoderForMaskedLM(config).eval()
    collator = MLMCollator(vocab, 0, 4, [0, 1, 2, 3, 4], MLMSettings(), seed=0)
    batch = collator(random_sequences(lengths, vocab, 0))
    with torch.no_grad(), exact_fp32():
        loss = model(**batch).loss.item()
    assert abs(loss - math.log(vocab)) < 0.5


def test_overfit_fixed_batch():
    torch.manual_seed(0)
    model = ArabicEncoderForMaskedLM(tiny_config()).train()
    batch = tiny_collator(seed=3)(random_sequences([32] * 8, seed=3))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for step in range(301):
        loss = model(**batch).loss
        if step == 300:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    assert loss.item() < 0.05


def test_reference_inputs_are_fixed():
    ids, mask = reference_inputs(tiny_config())
    assert ids.shape == (3, 48) and mask.sum(1).tolist() == [48, 31, 7]
    assert torch.equal(ids, reference_inputs(tiny_config())[0])


def load_notebook_model_def():
    path = ROOT / "notebooks" / "model_def.py"
    if not path.exists():
        pytest.skip("notebooks/model_def.py is not here")
    spec = importlib.util.spec_from_file_location("notebook_model_def", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["notebook_model_def"] = module
    spec.loader.exec_module(module)
    return module


def test_same_outputs_as_the_notebook_model():
    nb = load_notebook_model_def()
    torch.manual_seed(0)
    ours = ArabicEncoderForMaskedLM(tiny_config())
    theirs = nb.ArabicEncoderForMaskedLM(nb.ArabicEncoderConfig(**tiny_config().to_dict()))
    theirs.load_state_dict(ours.state_dict(), strict=True)
    ids, mask = reference_inputs(ours.config)
    a, b = reference_forward(ours, ids, mask), nb.reference_forward(theirs, ids, mask)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert [(k, tuple(v.shape)) for k, v in ours.state_dict().items()] == \
           [(k, tuple(v.shape)) for k, v in theirs.state_dict().items()]


@pytest.mark.parametrize("folder", ["stages/3_contrastive_weak/final_model", "stages/2_mlm_pretraining/final_model/encoder"])
def test_saved_models_load_strictly(folder):
    from pired.encoder.weights import load_encoder

    path = ROOT / folder
    if not (path / "model.safetensors").exists():
        pytest.skip(f"{folder} is not on this machine")
    model = load_encoder(path)
    diffs = check_reference_outputs(model, path, atol=1e-4)
    assert max(diffs.values()) < 1e-4
