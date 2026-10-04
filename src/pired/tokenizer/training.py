import dataclasses
import json
import math
from pathlib import Path

from tokenizers import Tokenizer, models, processors, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .corpus import iter_corpus_batches
from .normalization import build_decoder, build_normalizer, build_pre_tokenizer

N_BYTE_TOKENS = 256


def initial_alphabet(normalizer):
    candidates = "".join(chr(c) for c in range(0x21, 0x7F)) + "".join(chr(c) for c in range(0x0621, 0x064B))
    return sorted(set(normalizer.normalize_str(candidates)) - {" "})


def empty_tokenizer(cfg):
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = build_normalizer(cfg)
    tok.pre_tokenizer = build_pre_tokenizer(cfg)
    tok.decoder = build_decoder()
    return tok


def train_bpe(cfg, batches, n_batches=None, show_progress=True):
    tok = empty_tokenizer(cfg)
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size - N_BYTE_TOKENS,
        min_frequency=cfg.min_frequency,
        special_tokens=list(cfg.special_tokens),
        limit_alphabet=cfg.limit_alphabet,
        initial_alphabet=initial_alphabet(tok.normalizer),
        max_token_length=cfg.max_token_length,
        show_progress=show_progress,
    )
    tok.train_from_iterator(batches, trainer=trainer, length=n_batches)
    return tok


def train_bpe_on_corpus(cfg, manifest):
    n_docs = sum(c["train_docs"] for c in manifest["counts"].values())
    return train_bpe(cfg, iter_corpus_batches(Path(cfg.corpus_dir) / "train.jsonl"), math.ceil(n_docs / 1_000))


def add_byte_fallback(tok, special_tokens):
    spec = json.loads(tok.to_str())
    ordered = [t for t, _ in sorted(spec["model"]["vocab"].items(), key=lambda kv: kv[1])]
    if ordered[: len(special_tokens)] != list(special_tokens):
        raise ValueError("special tokens must come first")
    byte_tokens = [f"<0x{b:02X}>" for b in range(N_BYTE_TOKENS)]
    rest = ordered[len(special_tokens):]
    if set(byte_tokens) & set(rest):
        raise ValueError("byte tokens already in the vocabulary")
    vocab = {t: i for i, t in enumerate([*special_tokens, *byte_tokens, *rest])}
    spec["model"]["vocab"] = vocab
    spec["model"]["byte_fallback"] = True
    for added in spec["added_tokens"]:
        added["id"] = vocab[added["content"]]
    return Tokenizer.from_str(json.dumps(spec, ensure_ascii=False))


def add_post_processor(tok):
    cls_id, sep_id = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    return tok


def finish_tokenizer(raw, cfg):
    return add_post_processor(add_byte_fallback(raw, cfg.special_tokens))


def wrap_fast(backend, cfg):
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=cfg.model_max_length,
        padding_side="right",
        truncation_side="right",
        clean_up_tokenization_spaces=False,
        model_input_names=["input_ids", "attention_mask"],
    )


def save_tokenizer(backend, cfg, corpus_counts):
    out = Path(cfg.tokenizer_dir)
    out.mkdir(parents=True, exist_ok=True)
    wrap_fast(backend, cfg).save_pretrained(out)
    (out / "tokenizer_training_config.json").write_text(
        json.dumps({"config": dataclasses.asdict(cfg), "corpus": corpus_counts}, indent=2, default=str),
        encoding="utf-8",
    )
    reloaded = AutoTokenizer.from_pretrained(out)
    probe = "مرحبا بكم في 2024!"
    if reloaded(probe)["input_ids"] != backend.encode(probe).ids:
        raise AssertionError("the saved tokenizer encodes differently from the trained one")
    return reloaded
