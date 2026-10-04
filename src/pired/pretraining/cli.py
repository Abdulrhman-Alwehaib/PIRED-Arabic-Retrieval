import sys
from pathlib import Path

from transformers import AutoTokenizer

from ..common.cli import run
from ..common.device import setup_device
from ..common.io import read_json
from ..common.notify import make_notifier
from ..common.runlog import JsonlLogger, log, read_jsonl
from ..encoder.weights import load_masked_lm
from .benchmark import compile_benchmark, measure_with_oom_backoff, recommend_settings, sample_token_ids
from .config import make_config, make_corpus_config
from .corpus import CorpusBuilder
from .data import download_pretraining_data
from .probe import EXAMPLES, MaskFiller, heldout_accuracy, load_validation_rows
from .trainer import PretrainingRun, data_repo_id, load_tokenizer

CORPUS_STEPS = ("plan", "download", "hash", "dedup", "tokenize", "manifest", "verify", "push")


def corpus(project, args):
    cfg = make_corpus_config(project)
    cfg.finished_units_only = cfg.finished_units_only or args.finished_only
    steps = tuple(args.steps.split(",")) if args.steps else CORPUS_STEPS
    unknown = set(steps) - set(CORPUS_STEPS)
    if unknown:
        raise ValueError(f"unknown corpus steps {sorted(unknown)}; use {CORPUS_STEPS}")
    builder = CorpusBuilder(cfg, project.hf_token)
    builder.load_plan()
    log("\n" + builder.plan_summary())
    if "download" in steps:
        builder.download()
    if "hash" in steps:
        builder.hash_all()
    if "dedup" in steps:
        builder.dedup()
    if "tokenize" in steps:
        builder.tokenize_all()
    units = builder.finished_units() if cfg.finished_units_only else builder.units
    manifest = None
    if "manifest" in steps:
        manifest = builder.build_manifest(units)
        log("\n" + builder.manifest_summary(manifest))
    if {"verify", "push"} & set(steps):
        manifest = manifest or read_json(Path(cfg.out_dir) / "manifest.json")
    if "verify" in steps:
        log("structure OK; first sequence starts with:\n" + builder.verify(manifest))
    if "push" in steps:
        if cfg.push_to_hub and project.hf_token:
            builder.push(project.api, project.repo_id("pretraining-data"), manifest)
        else:
            log("push skipped (smoke run, push_to_hub off or no HF_TOKEN)")


def download_data(project, args):
    cfg = make_config(project)
    download_pretraining_data(cfg.data.local_dir, data_repo_id(cfg, project), project.hf_token,
                              cfg.data.delete_consumed_shards)


def benchmark(project, args):
    cfg = make_config(project)
    device = setup_device(require_gpu=True)
    tokenizer = load_tokenizer(cfg, project)
    docs_ids = sample_token_ids(cfg, tokenizer)
    logger = JsonlLogger(project.path("smoke_tests_dir") / "2_mlm_pretraining" / "train_log.jsonl",
                         notifier=make_notifier("MLM smoke run", cfg.notify.telegram))
    result = measure_with_oom_backoff(cfg, tokenizer, docs_ids, cfg.benchmark.micro_batch_size,
                                      warmup=cfg.benchmark.warmup_steps, timed=cfg.benchmark.timed_steps,
                                      compile_model=cfg.train.compile, device=device, logger=logger)
    log("\n" + recommend_settings(cfg, result, read_jsonl(logger.path)[-1]))
    if args.compile_benchmark:
        compile_benchmark(cfg, tokenizer, docs_ids, device)


def train(project, args):
    cfg = make_config(project)
    device = setup_device()
    if not cfg.smoke:
        download_pretraining_data(cfg.data.local_dir, data_repo_id(cfg, project), project.hf_token,
                                  cfg.data.delete_consumed_shards)
    pretraining = PretrainingRun(cfg, project, device)
    try:
        log("\n" + pretraining.describe())
        pretraining.resume()
        pretraining.train()
        pretraining.save_final()
    finally:
        pretraining.close()


def probe(project, args):
    cfg = make_config(project)
    device = setup_device()
    model_dir = Path(args.model) if args.model else Path(cfg.final_model_dir) / "mlm"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = load_masked_lm(model_dir).to(device).eval()
    filler = MaskFiller(model, tokenizer, device)
    for text in args.sentence or EXAMPLES:
        print(filler.describe(text), flush=True)
    local_rows = project.path("pretraining_data_dir") / "validation.npy"
    repo_id = None if local_rows.exists() else data_repo_id(cfg, project)
    rows = load_validation_rows(local_rows, repo_id, project.hf_token, args.rows)
    models = [("final" if not args.model else model_dir.name, model)]
    if args.compare:
        models.append((Path(args.compare).name, load_masked_lm(args.compare).to(device).eval()))
    for name, mlm in models:
        quiz, held = filler.quiz(model=mlm), heldout_accuracy(mlm, tokenizer, rows, device)
        log(f"{name:<28} quiz top-1 {quiz['top1']:>4.0%} top-5 {quiz['top5']:>4.0%} | held-out top-1 "
            f"{held['top1']:.1%} top-5 {held['top5']:.1%} loss {held['loss']:.3f} perplexity {held['perplexity']:.1f} "
            f"({held['masked_tokens']:,} masked tokens)")


def add_arguments(parser):
    parser.add_argument("--steps", default=None, help="corpus: comma-separated steps, default all: "
                                                      + ",".join(CORPUS_STEPS))
    parser.add_argument("--finished-only", action="store_true", help="corpus: only the units already tokenized")
    parser.add_argument("--compile-benchmark", action="store_true", help="benchmark: also torch.compile on vs off")
    parser.add_argument("--model", default=None, help="probe: an MLM folder (default final_model/mlm)")
    parser.add_argument("--compare", default=None, help="probe: a second MLM folder, e.g. the pre-decay checkpoint")
    parser.add_argument("--sentence", action="append", help="probe: a sentence with [MASK] (repeatable)")
    parser.add_argument("--rows", type=int, default=None, help="probe: held-out rows to use (default all)")


COMMANDS = {"corpus": corpus, "download-data": download_data, "benchmark": benchmark, "train": train, "probe": probe}


def main(argv=None):
    return run("pired pretraining", COMMANDS, argv, add_arguments)


if __name__ == "__main__":
    sys.exit(main())
