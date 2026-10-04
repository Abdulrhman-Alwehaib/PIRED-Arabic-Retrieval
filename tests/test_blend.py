import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from pired.blend.blend import BlendConfig, Blender, blend_state
from pired.encoder.modeling import ArabicEncoderModel
from pired.encoder.weights import check_reference_outputs, load_encoder, save_reference_outputs
from conftest import PASSAGE_PREFIX, QUERY_PREFIX, tiny_config


def test_blend_state_math():
    a = {"w": torch.tensor([0.0, 2.0]), "i": torch.tensor([1, 2])}
    b = {"w": torch.tensor([4.0, 6.0]), "i": torch.tensor([3, 4])}
    out = blend_state(a, b, 0.25)
    assert torch.equal(out["w"], torch.tensor([1.0, 3.0])) and torch.equal(out["i"], b["i"])


def save_model(folder, tokenizer, seed):
    torch.manual_seed(seed)
    model = ArabicEncoderModel(tiny_config(vocab_size=len(tokenizer))).eval()
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    pooling = {"pooling": "mean", "normalize": "l2", "query_prefix": QUERY_PREFIX, "passage_prefix": PASSAGE_PREFIX,
               "max_query_tokens": 64, "max_passage_tokens": 256, "step": seed}
    (Path(folder) / "pooling_config.json").write_text(json.dumps(pooling, ensure_ascii=False), encoding="utf-8")
    save_reference_outputs(model, folder)
    return model


def test_blender_end_to_end(tokenizer):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        save_model(tmp / "stage3", tokenizer, 1)
        save_model(tmp / "stage4" / "stage4-best", tokenizer, 2)
        cfg = BlendConfig(stage3_dir=tmp / "stage3", stage4_dir=tmp / "stage4", out_dir=tmp / "out", alphas=(0.5,),
                          push=False)
        project = SimpleNamespace(name="test", hf_token=None, api=None, repo_id=lambda *args, **kwargs: "x/y")
        saved = Blender(project, cfg, torch.device("cpu")).build_all()
        out = Path(saved[0.5]["folder"])
        a = load_file(str(tmp / "stage3" / "model.safetensors"))
        b = load_file(str(tmp / "stage4" / "stage4-best" / "model.safetensors"))
        mixed = load_file(str(out / "model.safetensors"))
        assert all(torch.equal(mixed[k], (0.5 * a[k].float() + 0.5 * b[k].float()).to(b[k].dtype)) for k in b)
        check_reference_outputs(load_encoder(out), out, atol=0.0)
        pooling = json.loads((out / "pooling_config.json").read_text(encoding="utf-8"))
        assert pooling["blend"]["alpha"] == 0.5 and pooling["checkpoint"] == "stage4-blend-0.5"
