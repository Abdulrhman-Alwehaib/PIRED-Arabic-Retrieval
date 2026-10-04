import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

EVAL_SET_BLOCKLIST = ("miracl", "tydi", "mteb")


@dataclass
class CorpusSource:
    name: str
    repo_id: str
    prefix: str
    share: float
    text_field: str = "text"


def default_corpus_sources():
    return [
        CorpusSource("fineweb2", "HuggingFaceFW/fineweb-2", "data/arb_Arab/train", share=0.60),
        CorpusSource("101b", "ClusterlabAi/101_billion_arabic_words_dataset", "data", share=0.38),
        CorpusSource("wiki", "wikimedia/wikipedia", "20231101.ar", share=0.02),
    ]


@dataclass
class CorpusConfig:
    target_tokens: float = 30e9
    margin: float = 0.10
    sources: list[CorpusSource] = field(default_factory=default_corpus_sources)
    seq_len: int = 1024
    unit_row_groups: int = 200
    min_chars: int = 100
    min_tokens: int = 32
    min_arabic_word_ratio: float = 0.5
    validation_every: int = 10_000
    max_validation_sequences: int = 4_096
    hash_workers: int = 8
    tokenize_workers: int = 1
    encode_batch_docs: int = 1000
    download_workers: int = 8
    seed: int = 1234
    tokenizer_dir: Path | None = None
    raw_dir: Path | None = None
    out_dir: Path | None = None
    push_to_hub: bool = True
    delete_raw_after_push: bool = False
    finished_units_only: bool = False
    remote_read: bool = False
    max_units_per_file: int | None = None


def make_corpus_config(project):
    cfg = CorpusConfig(tokenizer_dir=project.path("tokenizer_dir"), raw_dir=project.path("downloads_dir"),
                       out_dir=project.stage_dir("pretraining_data_dir"))
    if project.smoke:
        cfg = dataclasses.replace(cfg, target_tokens=30e6, unit_row_groups=2, max_units_per_file=1, remote_read=True,
                                  validation_every=50, max_validation_sequences=64, push_to_hub=False)
    for source in cfg.sources:
        if any(bad in source.repo_id.lower() for bad in EVAL_SET_BLOCKLIST):
            raise ValueError(f"{source.repo_id} looks like an evaluation set and must not be used for pretraining")
    if abs(sum(s.share for s in cfg.sources) - 1.0) > 1e-6:
        raise ValueError("source shares must add up to 1")
    return cfg


@dataclass
class ModelSettings:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 2048
    max_position_embeddings: int = 1024
    rope_theta: float = 10_000.0
    layer_norm_eps: float = 1e-5
    hidden_dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0
    initializer_range: float = 0.02


@dataclass
class MLMSettings:
    mlm_probability: float = 0.30
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    pad_to_multiple_of: int = 8


@dataclass
class TrainSettings:
    run_name: str = "mlm-base-v1"
    seq_len: int = 1024
    micro_batch_size: int = 32
    grad_accum_steps: int = 16
    total_steps: int = 20_690
    lr: float = 5e-4
    warmup_steps: int = 2_000
    lr_schedule: str = "wsd"
    decay_fraction: float = 0.2
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-6
    max_grad_norm: float = 1.0
    log_every: int = 10
    save_every_steps: int = 400
    keep_local_checkpoints: int = 2
    seed: int = 42
    compile: bool = True
    resume: bool = True
    init_from: str | None = None
    eval_every_steps: int = 1_000
    eval_sequences: int = 2_048


@dataclass
class DataSettings:
    local_dir: Path | None = None
    repo_id: str | None = None
    window_shards: int = 8
    delete_consumed_shards: bool = False


@dataclass
class HubSettings:
    push_checkpoints: bool = True
    checkpoint_repo: str | None = None
    keep_last: int = 3
    keep_every_steps: int = 10_000
    squash_history: bool = True


@dataclass
class NotifySettings:
    telegram: bool = True


@dataclass
class BenchmarkSettings:
    dataset_repo: str = "ClusterlabAi/101_billion_arabic_words_dataset"
    shard: str = "data/train-00469-of-00470.parquet"
    n_docs: int = 6_000
    micro_batch_size: int = 32
    warmup_steps: int = 30
    timed_steps: int = 270
    compile_bench_steps: int = 40


@dataclass
class PretrainingConfig:
    model: ModelSettings = field(default_factory=ModelSettings)
    mlm: MLMSettings = field(default_factory=MLMSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    data: DataSettings = field(default_factory=DataSettings)
    hub: HubSettings = field(default_factory=HubSettings)
    notify: NotifySettings = field(default_factory=NotifySettings)
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    tokenizer_dir: Path | None = None
    tokenizer_repo: str | None = None
    checkpoints_dir: Path | None = None
    runs_dir: Path | None = None
    final_model_dir: Path | None = None
    smoke: bool = False


def make_config(project):
    stage = project.stage_dir("mlm_stage_dir")
    cfg = PretrainingConfig(tokenizer_dir=project.path("tokenizer_dir"), checkpoints_dir=stage / "checkpoints",
                            runs_dir=stage / "logs", final_model_dir=stage / "final_model", smoke=project.smoke)
    cfg.data.local_dir = project.path("pretraining_data_dir")
    if project.smoke:
        cfg.train = dataclasses.replace(cfg.train, run_name="mlm-smoke", micro_batch_size=2, grad_accum_steps=2,
                                        total_steps=20, warmup_steps=4, save_every_steps=10, eval_every_steps=10,
                                        eval_sequences=8, log_every=5, compile=False)
        cfg.hub.push_checkpoints = False
        cfg.notify.telegram = False
        cfg.data.window_shards = 2
    return cfg
