from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelSources:
    stage3_dir: Path | None = None
    stage3_repo: str | None = None
    stage4_dir: Path | None = None
    stage4_repo: str | None = None
    stage4_best: str = "stage4-best"
    stage4_final: str = "stage4-final"
    blends: tuple[str, ...] = ("stage4-blend-0.3", "stage4-blend-0.5", "stage4-blend-0.7")
    tokenizer_fingerprint: str = "b490ba47ebc5a117"
    query_prefix: str = "استعلام: "
    passage_prefix: str = "نص: "
    max_query_tokens: int = 64
    max_passage_tokens: int = 256
    baseline: tuple[str, str, str, int] = ("intfloat/multilingual-e5-base", "query: ", "passage: ", 512)
    published: tuple[str, ...] = ("intfloat/multilingual-e5-large", "BAAI/bge-m3")


@dataclass
class EvalSettings:
    in_domain: tuple[str, ...] = ("MIRACLRetrieval", "MIRACLRetrievalHardNegatives", "MIRACLRetrievalHardNegatives.v2",
                                  "MrTidyRetrieval")
    skip: tuple[str, ...] = ()
    big_tasks: tuple[str, ...] = ("MIRACLRetrieval", "MrTidyRetrieval")
    arabic_subsets: tuple[str, ...] = ("default", "ar", "arabic", "ara", "ara-ara")
    top_k: int = 100
    batch_tokens: int = 65_536
    min_batch_tokens: int = 1_024
    stage3_miracl_ndcg10: float = 0.4761
    stage3_tolerance: float = 0.005
    nb04_results: Path | None = None
    nb04_mteb_version: str = "2.12.10"
    smoke_task: str = "MintakaRetrieval"
    smoke_queries: int = 50
    smoke_distractors: int = 300
    speed_texts: int = 400


@dataclass
class SpeedSettings:
    texts_dir: Path | None = None
    sources: tuple[str, ...] = ("news", "qa", "qa_mined", "spans", "wiki", "xlsum")
    per_source: int = 1000
    batch_size: int = 64
    max_length: int = 512
    repeats: int = 3
    cpu_threads: int = 10
    cpu_per_source: int = 200
    cpu_batch_size: int = 32


@dataclass
class EvaluationConfig:
    models: ModelSources = field(default_factory=ModelSources)
    eval: EvalSettings = field(default_factory=EvalSettings)
    speed: SpeedSettings = field(default_factory=SpeedSettings)
    stage_dir: Path | None = None
    results_dir: Path | None = None
    reports_dir: Path | None = None
    smoke: bool = False


def make_config(project):
    stage = project.stage_dir("evaluation_stage_dir")
    cfg = EvaluationConfig(stage_dir=stage, results_dir=stage / "results", reports_dir=stage / "reports",
                           smoke=project.smoke)
    contrastive = project.path("contrastive_weak_stage_dir")
    cfg.models.stage3_dir = contrastive / "final_model"
    cfg.models.stage4_dir = project.stage_dir("supervised_stage_dir") / "final_model"
    cfg.eval.nb04_results = contrastive / "logs" / "contrastive-weak-v1" / "eval_results.json"
    cfg.speed.texts_dir = contrastive / "data" / "final" / "validation"
    if project.smoke:
        cfg.eval.batch_tokens = 8_192
    return cfg
