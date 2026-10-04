import sys

from ..common.cli import run
from ..common.io import read_json
from ..common.runlog import log
from .checks import (Decontaminator, Part1State, check_step, ensure_checked_part1, save_part1,
                     write_data_report)
from .config import make_config
from .context import Stage
from .generation import Generator, build_synthetic_source, load_generated, synthetic_queries
from .mining import Miner, example_rows
from .queries import LABELED_BUILDERS, build_synthetic_passages, report_source
from .report import a100_estimates, stage_report, write_reports_and_push
from .run import Part3Data, TrainingRun, export_models
from .selftest import run_selftests

CHECKS = ("cleanup", "language", "mmarco_check", "length", "dedupe")


def make_stage(project, with_encoder=False):
    return Stage(project, make_config(project), with_encoder=with_encoder)


def queries(project, args):
    stage = make_stage(project)
    for builder in LABELED_BUILDERS.values():
        report_source(stage, builder(stage))
    passages = build_synthetic_passages(stage)
    log(f"{len(passages):,} passages for question generation: {dict(passages['domain'].value_counts())}")
    generator = Generator(stage)
    generator.prompt_test(passages)
    gen_stats = generator.run(passages)
    if gen_stats["passages"]:
        stage.timings["generation"] = gen_stats
    generated = load_generated(stage)
    generator.close()
    syn_q = synthetic_queries(stage, generated, passages)
    report_source(stage, build_synthetic_source(stage, syn_q, passages, generated, generator.backend.name,
                                                gen_stats.get("seconds", 0.0)))
    state = Part1State.from_sources(stage)
    for name in CHECKS:
        state.run_check(*check_step(stage, name))
    decontam_test = Decontaminator(stage).run(state)
    manifest = save_part1(state, {"loading": stage.loading, "generation": stage.timings.get("generation"),
                                  "generation_test": stage.timings.get("gen_test"), "decontam_test": decontam_test,
                                  "synthetic_passages": read_json(stage.syn / "passages.stats.json")})
    lines = [f"{'source':<11}{'train':>10}{'validation':>12}"]
    lines += [f"{s:<11}{v['train']:>10,}{v['validation']:>12,}" for s, v in manifest["queries"].items()]
    log("\n".join(lines) + f"\npassages referenced: {manifest['passages']:,}; saved to {stage.final1}")
    report = write_data_report(stage, manifest)
    log(report[report.find("## Funnel"):])
    stage.push_folder(stage.final1, stage.repo_id("queries"), "dataset")


def mine(project, args):
    stage = make_stage(project)
    ensure_checked_part1(stage)
    Miner(stage).run()
    log(example_rows(stage))


def prepare_part3(project):
    stage = make_stage(project, with_encoder=True)
    data = Part3Data(stage)
    return stage, data, data.before_validation()


def selftest(project, args):
    stage, data, _ = prepare_part3(project)
    run_selftests(stage, data.train, data.val, data.n_neg, data.stats1, data.stats2)


def train(project, args):
    stage, data, before_val = prepare_part3(project)
    tests = run_selftests(stage, data.train, data.val, data.n_neg, data.stats1, data.stats2)
    training = TrainingRun(stage, data)
    log("\n" + training.describe())
    training.estimate()
    training.resume()
    summary = training.train()
    exported = export_models(stage, training, data, tests, before_val)
    a100 = a100_estimates(stage, data)
    report = stage_report(stage, data, summary, exported, before_val, tests, a100)
    print(report, flush=True)
    write_reports_and_push(stage, report)


def all_parts(project, args):
    for step in (queries, mine, train):
        step(project, args)


COMMANDS = {"queries": queries, "mine": mine, "selftest": selftest, "train": train, "all": all_parts}


def main(argv=None):
    return run("pired supervised", COMMANDS, argv)


if __name__ == "__main__":
    sys.exit(main())
