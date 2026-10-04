from dataclasses import dataclass, field
from pathlib import Path

EVAL_SET_BLOCKLIST = ("miracl", "tydi", "mteb")


@dataclass
class SourceConfig:
    name: str
    repo_id: str
    shard_prefix: str
    n_train_docs: int
    n_shards: int
    n_heldout_docs: int
    text_field: str = "text"


def default_sources():
    return [
        SourceConfig("101b", "ClusterlabAi/101_billion_arabic_words_dataset", "data/",
                     n_train_docs=500_000, n_shards=80, n_heldout_docs=2_000),
        SourceConfig("wiki", "wikimedia/wikipedia", "20231101.ar/",
                     n_train_docs=140_000, n_shards=7, n_heldout_docs=1_000),
    ]


def smoke_sources():
    return [
        SourceConfig("101b", "ClusterlabAi/101_billion_arabic_words_dataset", "data/",
                     n_train_docs=3_000, n_shards=2, n_heldout_docs=200),
        SourceConfig("wiki", "wikimedia/wikipedia", "20231101.ar/",
                     n_train_docs=1_000, n_shards=1, n_heldout_docs=100),
    ]


@dataclass
class TokenizerConfig:
    vocab_size: int = 64_000
    special_tokens: tuple[str, ...] = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
    min_frequency: int = 2
    limit_alphabet: int = 500
    max_token_length: int | None = 24
    model_max_length: int = 1024
    nfkc: bool = True
    remove_invisible: bool = True
    remove_tatweel: bool = True
    remove_tashkeel: bool = True
    unify_alef: bool = True
    arabic_digits_to_ascii: bool = True
    unify_alef_maqsura: bool = False
    unify_teh_marbuta: bool = False
    normalize_whitespace: bool = True
    individual_digits: bool = True
    split_punctuation: bool = True
    sources: list[SourceConfig] = field(default_factory=default_sources)
    heldout_every: int = 50
    min_doc_chars: int = 20
    download_workers: int = 8
    seed: int = 1234
    corpus_dir: Path | None = None
    tokenizer_dir: Path | None = None
    push_to_hub: bool = True
    reference_tokenizer: str = "intfloat/multilingual-e5-base"
    fertility_words_per_category: int = 100_000


def check_sources(sources):
    for source in sources:
        if any(bad in source.repo_id.lower() for bad in EVAL_SET_BLOCKLIST):
            raise ValueError(f"{source.repo_id} looks like an evaluation set and must not be used for tokenizer training")


def make_config(project):
    cfg = TokenizerConfig(corpus_dir=project.stage_dir("tokenizer_corpus_dir"),
                          tokenizer_dir=project.stage_dir("tokenizer_dir"))
    if project.smoke:
        cfg.vocab_size = 4_000
        cfg.sources = smoke_sources()
        cfg.push_to_hub = False
        cfg.fertility_words_per_category = 5_000
    check_sources(cfg.sources)
    return cfg
