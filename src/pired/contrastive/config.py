import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

EVAL_SETS_FORBIDDEN = ("miracl", "tydi", "mteb")


@dataclass
class SourceSettings:
    name: str
    description: str
    target: int | None
    raw: int | None
    license: str


def default_sources():
    return [
        SourceSettings("news", "Abu El-Khair news corpus: headline -> first paragraph", 4_000_000, 6_000_000,
                       "unknown (the original card says 'unknown'; the parquet mirror states none)"),
        SourceSettings("wiki", "Arabic Wikipedia: title -> lead; 'title heading' -> section text", 1_500_000,
                       2_400_000, "CC BY-SA 3.0 / GFDL"),
        SourceSettings("xlsum", "XL-Sum Arabic, train: title -> summary; summary -> start of the article", 70_000,
                       None, "CC BY-NC-SA 4.0 (non-commercial)"),
        SourceSettings("qa", "Open-ArabicaQA train: question -> passage; Aya, Arabic varieties: prompt -> answer",
                       None, None, "MIT (ArabicaQA); Apache-2.0 (Aya)"),
        SourceSettings("qa_mined", "FineWeb-2 arb_Arab: sentence ending in ؟ -> the paragraph after it",
                       2_500_000, 4_000_000, "ODC-By 1.0 (FineWeb-2; Common Crawl terms of use apply)"),
        SourceSettings("spans", "101B Arabic Words: short span -> longer span of the same document", 5_000_000,
                       8_000_000, "Apache-2.0 (dataset card)"),
    ]


@dataclass
class DataSettings:
    work_dir: Path | None = None
    raw_dir: Path | None = None
    sources: list[SourceSettings] = field(default_factory=default_sources)
    smoke_rows: int = 2_000
    shard_rows: int = 100_000
    workers: int = field(default_factory=lambda: os.cpu_count() or 1)
    seed: int = 1234
    max_text_chars: int = 3_000
    news_repo: str = "MohamedRashad/arabic-billion-words"
    news_rg_per_unit: int = 50
    wiki_repo: str = "wikimedia/wikipedia"
    wiki_prefix: str = "20231101.ar"
    wiki_rg_per_unit: int = 20
    xlsum_repo: str = "csebuetnlp/xlsum"
    xlsum_revision: str = "refs/convert/parquet"
    xlsum_file: str = "arabic/train/0000.parquet"
    arabicaqa_repo: str = "abdoelsayed/Open-ArabicaQA"
    arabicaqa_train: str = "human-annotated/train-open.jsonl"
    arabicaqa_eval: tuple[str, ...] = ("human-annotated/test-open.jsonl", "human-annotated/val-open.jsonl")
    arabicaqa_mrc_repo: str = "abdoelsayed/ArabicaQA"
    arabicaqa_mrc_train: str = "MRC/train.json"
    aya_repo: str = "CohereLabs/aya_dataset"
    aya_file: str = "data/train-00000-of-00001.parquet"
    fineweb2_repo: str = "HuggingFaceFW/fineweb-2"
    fineweb2_prefix: str = "data/arb_Arab/train"
    fineweb2_rg_per_unit: int = 100
    b101_repo: str = "ClusterlabAi/101_billion_arabic_words_dataset"
    b101_prefix: str = "data"
    spans_corpus: str = "101b"
    news_min_words: int = 25
    wiki_min_words: int = 20
    qa_mined_words: tuple[int, int] = (3, 30)
    qa_mined_min_answer_words: int = 20
    qa_mined_max_per_doc: int = 2
    arabicaqa_window_chars: int = 1_200
    span_query_tokens: tuple[int, int] = (16, 64)
    span_passage_tokens: tuple[int, int] = (64, 256)


EVAL_QUERY_FILES = {
    "MIRACL-ar": ("miracl/miracl", [f"miracl-v1.0-ar/topics/topics.miracl-v1.0-ar-{s}.tsv"
                                    for s in ("dev", "test-a", "test-b", "train")]),
    "Mr.TyDi-ar": ("castorini/mr-tydi", [f"mrtydi-v1.1-arabic/ir-format-data/topics.{s}.txt"
                                         for s in ("train", "dev", "test")]),
    "Open-ArabicaQA": ("abdoelsayed/Open-ArabicaQA", ["human-annotated/test-open.jsonl",
                                                      "human-annotated/val-open.jsonl"]),
}

BOILERPLATE_TITLES = (
    "اتصل بنا", "اتصل بنا الان", "الرئيسية", "الصفحة الرئيسية", "الرئيسة", "من نحن", "عن الموقع", "عن الشركة",
    "سياسة الخصوصية", "شروط الاستخدام", "اتفاقية الاستخدام", "الشروط والاحكام", "خريطة الموقع", "تسجيل الدخول",
    "تسجيل دخول", "انشاء حساب", "اعلن معنا", "اعلن هنا", "للاعلان", "المزيد", "اقرا المزيد", "اقرا ايضا",
    "شاهد ايضا", "تابعنا", "تابعونا", "جميع الحقوق محفوظة", "حقوق النشر", "الارشيف", "بحث", "القائمة", "روابط",
    "روابط سريعة", "النشرة البريدية", "اشترك", "مقالات ذات صلة", "مواضيع ذات صلة", "اخبار ذات صلة", "تعليقات",
    "التعليقات", "اضف تعليق", "اترك تعليقا", "الاكثر قراءة", "الاكثر مشاهدة", "اخر الاخبار", "احدث الاخبار",
    "اخبار عاجلة", "عاجل", "صور", "فيديو", "مشاركة", "شارك", "طباعة", "مراجع", "المراجع", "وصلات خارجية",
    "انظر ايضا", "الاسئلة الشائعة")


@dataclass
class FilterSettings:
    min_query_tokens: int = 3
    min_passage_tokens: int = 20
    max_query_tokens: int = 64
    max_passage_tokens: int = 256
    min_arabic_letters: float = 0.70
    boilerplate_titles: tuple[str, ...] = BOILERPLATE_TITLES
    max_query_repeats: int = 5
    max_jaccard: float = 0.8
    minhash_ngram: int = 5
    minhash_perms: int = 64
    minhash_bands: int = 16
    near_dup_jaccard: float = 0.8
    decontam_ngram: int = 13
    decontam_min_words: int = 6
    bge_model: str = "BAAI/bge-m3"
    bge_negatives: int = 99
    bge_keep_top: int = 3
    bge_shard_rows: int = 200_000
    bge_query_max_len: int = 128
    bge_passage_max_len: int = 512
    bge_batch_tokens: int = 131_072
    max_share: float = 0.40
    val_pairs_per_source: int = 5_000


@dataclass
class ModelSettings:
    mlm_checkpoint: Path | None = None
    hub_repo: str | None = None
    hub_weights: str = "checkpoints/mlm-base-v1/step-00020690"
    hub_extras: str = "encoder"
    tokenizer_fingerprint: str = "b490ba47ebc5a117"
    use_prefixes: bool = True
    query_prefix: str = "استعلام: "
    passage_prefix: str = "نص: "


@dataclass
class TrainSettings:
    run_name: str = "contrastive-weak-v1"
    batch_size: int = 4096
    temperature: float = 0.02
    bidirectional: bool = True
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    warmup_fraction: float = 0.05
    max_grad_norm: float = 1.0
    max_steps: int | None = None
    chunk_tokens: int = 32_768
    log_every: int = 10
    eval_every: int = 250
    save_every: int = 1_000
    keep_local_checkpoints: int = 2
    seed: int = 42
    resume: bool = True
    compile: bool = False
    collapse_cosine: float = 0.95
    collapse_watch: float = 0.85
    val_random_pairs: int = 20_000
    eval_batch_tokens: int = 131_072


@dataclass
class HubSettings:
    push: bool = True
    data_repo: str | None = None
    checkpoint_repo: str | None = None
    model_repo: str | None = None
    keep_last: int = 3


@dataclass
class NotifySettings:
    telegram: bool = True


@dataclass
class EvalSettings:
    tasks: tuple[str, ...] = ("MIRACLRetrieval", "MrTidyRetrieval")
    languages: tuple[str, ...] = ("ara",)
    batch_tokens: int = 131_072
    baseline: str = "intfloat/multilingual-e5-base"
    baseline_prefixes: tuple[str, str] = ("query: ", "passage: ")
    baseline_max_len: int = 512
    smoke_queries: int = 200
    smoke_distractors: int = 5_000


@dataclass
class EstimateSettings:
    h100_train_tflops: float = 250.0
    h100_infer_tflops: float = 350.0
    cpu_cores: int = 32
    download_mb_per_s: float = 200.0


@dataclass
class ContrastiveConfig:
    data: DataSettings = field(default_factory=DataSettings)
    filters: FilterSettings = field(default_factory=FilterSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    hub: HubSettings = field(default_factory=HubSettings)
    notify: NotifySettings = field(default_factory=NotifySettings)
    eval: EvalSettings = field(default_factory=EvalSettings)
    estimate: EstimateSettings = field(default_factory=EstimateSettings)
    runs_dir: Path | None = None
    checkpoints_dir: Path | None = None
    final_dir: Path | None = None
    reports_dir: Path | None = None
    smoke: bool = False


def check_not_eval_sets(cfg):
    d = cfg.data
    for repo in (d.news_repo, d.wiki_repo, d.xlsum_repo, d.arabicaqa_repo, d.arabicaqa_mrc_repo, d.aya_repo,
                 d.fineweb2_repo, d.b101_repo):
        if any(bad in repo.lower() for bad in EVAL_SETS_FORBIDDEN):
            raise ValueError(f"{repo} looks like an evaluation set")


def make_config(project):
    stage = project.stage_dir("contrastive_weak_stage_dir")
    cfg = ContrastiveConfig(runs_dir=stage / "logs", checkpoints_dir=stage / "checkpoints",
                            final_dir=stage / "final_model", reports_dir=stage / "reports", smoke=project.smoke)
    cfg.data.work_dir = stage / "data"
    cfg.data.raw_dir = project.path("downloads_dir")
    cfg.model.mlm_checkpoint = project.path("mlm_stage_dir") / "final_model" / "mlm"
    cfg.hub.push = not project.smoke
    if project.smoke:
        cfg.filters.val_pairs_per_source = 100
        cfg.filters.bge_shard_rows = 1_000_000
        cfg.filters.bge_batch_tokens = 16_384
        cfg.train = dataclasses.replace(cfg.train, run_name="contrastive-weak-smoke", batch_size=64, max_steps=20,
                                        chunk_tokens=2_048, log_every=5, eval_every=10, save_every=10,
                                        val_random_pairs=2_000, eval_batch_tokens=16_384)
        cfg.eval.batch_tokens = 16_384
    check_not_eval_sets(cfg)
    return cfg
