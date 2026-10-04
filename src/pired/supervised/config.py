import dataclasses
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SourceSettings:
    name: str
    description: str
    domain: str
    human: bool
    target: int | None
    raw: int | None
    license: str


def default_sources():
    return [
        SourceSettings("miracl", "MIRACL Arabic train: question -> judged relevant passages (+ its judged non-relevant ones)",
                       "wiki", True, None, None, "Apache-2.0"),
        SourceSettings("mrtydi", "Mr.TyDi Arabic train: question -> relevant passage", "wiki", True, None, None,
                       "Apache-2.0"),
        SourceSettings("tydiqa", "TyDi QA Arabic train, gold passage task: question -> the gold passage", "wiki", True,
                       None, None, "Apache-2.0"),
        SourceSettings("arabicaqa", "Open-ArabicaQA train: question -> the paragraph holding the answer", "wiki", True,
                       None, None, "MIT"),
        SourceSettings("mmarco", "mMARCO Arabic train (MS MARCO, machine translated): query -> relevant passage",
                       "mmarco", False, 400_000, 600_000,
                       "mMARCO: Apache-2.0; MS MARCO terms: non-commercial research use only"),
        SourceSettings("synthetic", "questions written by Qwen3.6 for Wikipedia (70%) and news (30%) passages",
                       "wiki+news", False, 500_000, None,
                       "questions generated with Qwen3.6 (Apache-2.0); passages: Wikipedia via MIRACL (CC BY-SA 3.0), "
                       "Abu El-Khair news (license unknown)"),
    ]


@dataclass
class DataSettings:
    work_dir: Path | None = None
    raw_dir: Path | None = None
    sources: list[SourceSettings] = field(default_factory=default_sources)
    seed: int = 1234
    smoke_queries: int = 500
    miracl_repo: str = "miracl/miracl"
    miracl_dir: str = "miracl-v1.0-ar"
    miracl_corpus_repo: str = "miracl/miracl-corpus"
    miracl_corpus_dir: str = "miracl-corpus-v1.0-ar"
    mrtydi_repo: str = "castorini/mr-tydi"
    mrtydi_dir: str = "mrtydi-v1.1-arabic"
    tydiqa_repo: str = "google-research-datasets/tydiqa"
    tydiqa_train: str = "secondary_task/train-00000-of-00001.parquet"
    tydiqa_dev: str = "secondary_task/validation-00000-of-00001.parquet"
    arabicaqa_repo: str = "abdoelsayed/Open-ArabicaQA"
    arabicaqa_train: str = "human-annotated/train-open.jsonl"
    arabicaqa_eval: tuple[str, ...] = ("human-annotated/test-open.jsonl", "human-annotated/val-open.jsonl")
    arabicaqa_mrc_repo: str = "abdoelsayed/ArabicaQA"
    arabicaqa_mrc_train: str = "MRC/train.json"
    answer_window_chars: int = 1_200
    mmarco_repo: str = "unicamp-dl/mmarco"
    mmarco_queries: str = "data/google/queries/train/arabic_queries.train.tsv"
    mmarco_collection: str = "data/google/collections/arabic_collection.tsv"
    msmarco_qrels_repo: str = "BeIR/msmarco-qrels"
    msmarco_qrels_file: str = "train.tsv"
    mmarco_smoke_bytes: int = 12_000_000
    news_repo: str | None = None
    news_prefix: str = "train/news"
    shard_rows: int = 250_000


@dataclass
class GenSettings:
    n_passages: int = 500_000
    wiki_share: float = 0.70
    min_passage_tokens: int = 20
    max_per_article: int = 2
    backend: str = "server"
    model: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    smoke_model: str = "Qwen/Qwen3-1.7B"
    endpoint: str = "http://localhost:8000/v1"
    api_model: str | None = None
    api_key_env: str = "QWEN_API_KEY"
    extra_body: dict = field(default_factory=lambda: {"chat_template_kwargs": {"enable_thinking": False}})
    json_mode: str = "schema"
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 200
    concurrency: int = 256
    timeout_s: float = 300.0
    shard_passages: int = 5_000
    max_error_share: float = 0.05
    prompt_test_passages: int = 50
    min_parsed_share: float = 0.8
    vllm_env: str = "/opt/vllm-env"
    vllm_args: str = "--max-model-len 4096 --gpu-memory-utilization 0.88 --max-num-seqs 256"
    vllm_port: int = 8000
    question_words: tuple[str, ...] = ("ما", "من", "متى", "أين", "كيف", "لماذا", "هل", "كم")
    words_question: tuple[int, int] = (3, 40)
    words_search: tuple[int, int] = (2, 6)
    copy_ngram: int = 5
    banned_words: tuple[str, ...] = ("النص", "الفقرة", "المقال")
    max_questions_per_passage: int = 2


@dataclass
class FilterSettings:
    max_query_tokens: int = 64
    max_passage_tokens: int = 256
    min_query_words: int = 2
    min_passage_tokens: int = 20
    min_passage_tokens_human: int = 5
    min_arabic_letters: float = 0.70
    mmarco_latin_run: int = 4
    mmarco_repeat_run: int = 4
    mmarco_min_distinct: float = 0.30
    mmarco_min_passage_words: int = 8
    decontam_ngram: int = 13
    decontam_min_words: int = 6
    decontam_mteb_skip: tuple[str, ...] = ()
    decontam_mteb_subsets: tuple[str, ...] = ("default", "ar", "arabic", "ara")
    val_queries_per_source: int = 2_000
    val_queries_max: dict = field(default_factory=lambda: {"miracl": 300})
    max_share: float = 0.65
    mmarco_max_share: float = 0.40
    repeats_human: int = 2


@dataclass
class MiningSettings:
    top_k: int = 50
    rerank_top: int = 15
    reranker: str = "BAAI/bge-reranker-v2-m3"
    reranker_max_len: int = 512
    rerank_batch_tokens: int = 65_536
    rerank_shard_queries: int = 50_000
    false_negative_ratio: float = 0.95
    consistency_top: int = 3
    consistency_top_by_source: dict = field(default_factory=lambda: {"mmarco": 5})
    n_negatives: int = 7
    max_positives_scored: int = 5
    encode_batch_tokens: int = 262_144
    news_pool_size: int = 1_000_000
    pool_shard_rows: int = 500_000
    search_query_chunk: int = 8_192
    val_random_passages: int = 100_000
    val_candidates: int = 50
    rerank_caps: dict = field(default_factory=lambda: {"mmarco": 400_000, "synthetic": 800_000})


@dataclass
class ModelSettings:
    stage3_dir: Path | None = None
    stage3_repo: str | None = None
    tokenizer_fingerprint: str = "b490ba47ebc5a117"
    query_prefix: str = "استعلام: "
    passage_prefix: str = "نص: "


@dataclass
class TrainSettings:
    run_name: str = "supervised-v1"
    batch_size: int = 512
    temperature: float = 0.02
    bidirectional: bool = False
    max_neg_score: float | None = 0.75
    lr: float = 2e-5
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    warmup_fraction: float = 0.05
    max_grad_norm: float = 1.0
    max_steps: int | None = None
    chunk_tokens: int = 32_768
    log_every: int = 10
    eval_every: int = 200
    save_every_minutes: float = 30.0
    save_every_steps: int | None = None
    keep_local_checkpoints: int = 2
    seed: int = 42
    resume: bool = True
    compile: bool = False
    collapse_cosine: float = 0.95
    collapse_watch: float = 0.85
    eval_batch_tokens: int = 131_072
    random_pairs: int = 20_000


@dataclass
class HubSettings:
    push: bool = True
    queries_repo: str | None = None
    pairs_repo: str | None = None
    checkpoint_repo: str | None = None
    model_repo: str | None = None
    best_name: str = "stage4-best"
    final_name: str = "stage4-final"
    keep_last: int = 3


@dataclass
class NotifySettings:
    telegram: bool = True


@dataclass
class EstimateSettings:
    a100_encode_tflops: float = 30.0
    a100_train_tflops: float = 35.0
    a100_rerank_tflops: float = 45.0
    a100_gen_output_tokens_per_s: float = 2_500.0
    a100_gen_prompt_tokens_per_s: float = 20_000.0


@dataclass
class SupervisedConfig:
    data: DataSettings = field(default_factory=DataSettings)
    gen: GenSettings = field(default_factory=GenSettings)
    filters: FilterSettings = field(default_factory=FilterSettings)
    mining: MiningSettings = field(default_factory=MiningSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    hub: HubSettings = field(default_factory=HubSettings)
    notify: NotifySettings = field(default_factory=NotifySettings)
    estimate: EstimateSettings = field(default_factory=EstimateSettings)
    runs_dir: Path | None = None
    checkpoints_dir: Path | None = None
    final_dir: Path | None = None
    reports_dir: Path | None = None
    smoke: bool = False


def make_config(project):
    stage = project.stage_dir("supervised_stage_dir")
    cfg = SupervisedConfig(runs_dir=stage / "logs", checkpoints_dir=stage / "checkpoints", final_dir=stage / "final_model",
                           reports_dir=stage / "reports", smoke=project.smoke)
    cfg.data.work_dir = stage / "data"
    cfg.data.raw_dir = project.path("downloads_dir")
    cfg.model.stage3_dir = project.path("contrastive_weak_stage_dir") / "final_model"
    cfg.hub.push = not project.smoke
    if project.smoke:
        cfg.gen = dataclasses.replace(cfg.gen, backend="transformers", n_passages=20, prompt_test_passages=20,
                                      shard_passages=10, max_tokens=160)
        cfg.filters.val_queries_per_source = 50
        cfg.mining = dataclasses.replace(cfg.mining, encode_batch_tokens=16_384, rerank_batch_tokens=8_192,
                                         rerank_shard_queries=1_000, news_pool_size=4_000, pool_shard_rows=5_000,
                                         val_random_passages=2_000, search_query_chunk=1_024)
        cfg.train = dataclasses.replace(cfg.train, run_name="supervised-smoke", batch_size=16, max_steps=20,
                                        chunk_tokens=4_096, log_every=5, eval_every=10, save_every_minutes=0.0,
                                        save_every_steps=10, eval_batch_tokens=16_384, random_pairs=2_000)
    return cfg
