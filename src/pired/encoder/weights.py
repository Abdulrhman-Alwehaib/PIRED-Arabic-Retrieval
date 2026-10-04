from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from ..common.device import exact_fp32
from ..common.hub import missing_files
from .configuration import ArabicEncoderConfig
from .modeling import ArabicEncoderForMaskedLM, ArabicEncoderModel, mean_pool

REFERENCE_FILE = "reference_outputs.safetensors"
MODEL_FILES = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "pooling_config.json",
               REFERENCE_FILE)


def read_state(directory):
    files = sorted(Path(directory).glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors in {directory}")
    state = {}
    for f in files:
        state.update(load_file(str(f)))
    return state


def load_weights_strict(model: nn.Module, directory) -> nn.Module:
    state = read_state(directory)
    tied = getattr(model, "all_tied_weights_keys", None) or getattr(model, "_tied_weights_keys", None) or {}
    for target, source in (tied.items() if isinstance(tied, dict) else []):
        if target not in state and source in state:
            state[target] = state[source]
    model.load_state_dict(state, strict=True)
    return model


def load_encoder(directory) -> ArabicEncoderModel:
    return load_weights_strict(ArabicEncoderModel(ArabicEncoderConfig.from_pretrained(directory)), directory)


def load_masked_lm(directory) -> ArabicEncoderForMaskedLM:
    return load_weights_strict(ArabicEncoderForMaskedLM(ArabicEncoderConfig.from_pretrained(directory)), directory)


def load_encoder_without_head(mlm_dir):
    state = read_state(mlm_dir)
    config = ArabicEncoderConfig.from_pretrained(mlm_dir)
    with torch.device("meta"):
        head_keys = {k for k in ArabicEncoderForMaskedLM(config).state_dict() if k.startswith("head.")}
    encoder_state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    skipped = sorted(k for k in state if not k.startswith("model."))
    if not_head := [k for k in skipped if k not in head_keys]:
        raise ValueError(f"checkpoint keys that are neither encoder nor MLM head: {not_head}")
    encoder = ArabicEncoderModel(config)
    encoder.load_state_dict(encoder_state, strict=True)
    return encoder, skipped


def reference_inputs(config: ArabicEncoderConfig) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(20260923)
    first_regular = max(config.pad_token_id, config.cls_token_id, config.sep_token_id, config.mask_token_id) + 1
    lengths = [48, 31, 7]
    input_ids = torch.full((len(lengths), max(lengths)), config.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, n in enumerate(lengths):
        body = torch.randint(first_regular, config.vocab_size, (n - 2,), generator=g)
        input_ids[i, :n] = torch.cat([torch.tensor([config.cls_token_id]), body, torch.tensor([config.sep_token_id])])
        attention_mask[i, :n] = 1
    return input_ids, attention_mask


@torch.no_grad()
def reference_forward(model, input_ids, attention_mask) -> dict[str, torch.Tensor]:
    encoder = model.model if isinstance(model, ArabicEncoderForMaskedLM) else model
    device = next(model.parameters()).device
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    was_training = model.training
    model.eval()
    with exact_fp32():
        hidden = encoder(input_ids, attention_mask).last_hidden_state
        out = {"last_hidden_state": hidden,
               "embedding": F.normalize(mean_pool(hidden, attention_mask), p=2, dim=-1)}
        if isinstance(model, ArabicEncoderForMaskedLM):
            out["mlm_logits_first4"] = model.head(hidden[:, :4])
    model.train(was_training)
    return {k: v.float().cpu().contiguous() for k, v in out.items()}


def save_reference_outputs(model, directory) -> Path:
    input_ids, attention_mask = reference_inputs(model.config)
    path = Path(directory) / REFERENCE_FILE
    save_file({"input_ids": input_ids, "attention_mask": attention_mask,
               **reference_forward(model, input_ids, attention_mask)}, str(path))
    return path


def check_reference_outputs(model, directory, atol: float = 1e-4) -> dict[str, float]:
    ref = load_file(str(Path(directory) / REFERENCE_FILE))
    out = reference_forward(model, ref["input_ids"], ref["attention_mask"])
    mask = ref["attention_mask"].bool()
    diffs = {}
    for key, value in out.items():
        expected = ref[key]
        if key == "last_hidden_state":
            value, expected = value[mask], expected[mask]
        diffs[key] = (value - expected).abs().max().item()
    bad = {k: d for k, d in diffs.items() if d > atol}
    if bad:
        raise AssertionError(f"reference outputs differ: {bad}")
    return diffs


def has_model_files(folder, names=MODEL_FILES):
    return not missing_files(folder, names)


def ensure_model_folder(folder, repo_id, token, subfolder=None, names=MODEL_FILES):
    folder = Path(folder)
    target = folder / subfolder if subfolder else folder
    missing = missing_files(target, names)
    if missing:
        patterns = [f"{subfolder}/{name}" if subfolder else name for name in missing]
        snapshot_download(repo_id, allow_patterns=patterns, local_dir=folder, token=token)
    return target


def config_differences(config_a, config_b):
    meta = {"transformers_version", "_name_or_path", "architectures", "dtype", "torch_dtype"}
    a = {k: v for k, v in config_a.to_dict().items() if k not in meta}
    b = {k: v for k, v in config_b.to_dict().items() if k not in meta}
    return {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
