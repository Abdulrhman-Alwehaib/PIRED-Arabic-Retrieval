import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pired.encoder.configuration import ArabicEncoderConfig
from pired.encoder.embedding import Embedder, load_tokenizer, word_start_table
from pired.encoder.modeling import ArabicEncoderModel

TINY = dict(vocab_size=1000, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128,
            max_position_embeddings=128, pad_token_id=0, cls_token_id=2, sep_token_id=3, mask_token_id=4)
TOKENIZER_DIR = ROOT / "stages" / "1_tokenizer" / "tokenizer"
QUERY_PREFIX, PASSAGE_PREFIX = "استعلام: ", "نص: "


def tiny_config(**overrides):
    return ArabicEncoderConfig(**{**TINY, **overrides})


def random_sequences(lengths, vocab_size=1000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [[2, *torch.randint(5, vocab_size, (n - 2,), generator=g).tolist(), 3] for n in lengths]


def pad_batch(sequences, pad_id=0):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(sequences):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids, mask


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def tokenizer():
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("the stage-1 tokenizer is not on this machine")
    return load_tokenizer(TOKENIZER_DIR)


def make_text_stage(tokenizer, device, **train):
    torch.manual_seed(0)
    model = ArabicEncoderModel(tiny_config(vocab_size=len(tokenizer), max_position_embeddings=512)).to(device)
    stage = SimpleNamespace(tokenizer=tokenizer, device=device, q_prefix=QUERY_PREFIX, p_prefix=PASSAGE_PREFIX,
                            max_q=64, max_p=256, pad_id=tokenizer.pad_token_id,
                            is_word_start=word_start_table(tokenizer), source_names=["a", "b"],
                            cfg=SimpleNamespace(train=SimpleNamespace(**train)))
    stage.n_q_prefix = len(tokenizer(QUERY_PREFIX, add_special_tokens=False)["input_ids"])
    stage.n_p_prefix = len(tokenizer(PASSAGE_PREFIX, add_special_tokens=False)["input_ids"])
    stage.encoder = model

    def make_embedder(m):
        return Embedder(m, tokenizer, device, QUERY_PREFIX, PASSAGE_PREFIX, 64, 256)

    stage.make_embedder = make_embedder
    stage.embedder = make_embedder(model)
    stage.tokenize_texts = stage.embedder.tokenize
    stage.normalize = tokenizer.backend_tokenizer.normalizer.normalize_str
    return stage, model
