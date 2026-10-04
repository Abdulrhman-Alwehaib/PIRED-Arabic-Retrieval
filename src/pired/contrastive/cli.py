import sys

from ..common.cli import run
from ..common.io import read_json
from ..common.runlog import log
from .config import make_config
from .consistency import format_consistency, run_consistency
from .context import Stage
from .decontam import run_decontam
from .dedupe import run_dedupe
from .mix import format_mix, manual_review, push_final_data, run_final_mix, write_data_report
from .report import estimate_real_run, stage_report
from .run import TrainingRun, evaluate_after, evaluate_before, prepare_training_data, training_summary
from .selftest import run_selftests
from .sources import build_all_sources, raw_summary
from .steps import format_cleanup_changes, report_step, run_basic_filters, run_cleanup


def make_stage(project, with_encoder=False):
    stage = Stage(project, make_config(project))
    if with_encoder:
        encoder, skipped = stage.load_encoder()
        stage.skipped_keys = skipped
        log(f"encoder: {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M parameters, strict load; skipped "
            f"{len(skipped)} MLM head keys")
    else:
        stage.load_text_tools()
    return stage


def sources(project, args, stage=None):
    stage = stage or make_stage(project)
    build_all_sources(stage)
    summary = raw_summary(stage)
    total = sum(s["pairs"] for s in summary.values())
    lines = [f"{'source':<10}{'raw pairs':>12}{'share':>8}   kinds"]
    lines += [f"{name:<10}{s['pairs']:>12,}{s['pairs'] / max(total, 1):>8.1%}   {s['kinds']}" for name, s in summary.items()]
    log("\n".join(lines + [f"{'total':<10}{total:>12,}"]))
    return stage


def filter_pairs(project, args, stage=None):
    stage = stage or make_stage(project)
    cleanup = report_step(stage, "cleanup", run_cleanup(stage))
    log(format_cleanup_changes(cleanup))
    run_basic_filters(stage)
    report_step(stage, "dedupe", run_dedupe(stage))
    report_step(stage, "decontam", run_decontam(stage))
    consistency = report_step(stage, "consistency", run_consistency(stage))
    log(format_consistency(stage, consistency))
    return stage


def mix(project, args, stage=None):
    stage = stage or make_stage(project)
    manifest = run_final_mix(stage)
    log("\n" + format_mix(stage, manifest))
    log(manual_review(stage))
    write_data_report(stage, manifest, read_json(stage.steps_dir / "consistency" / "stats.json"))
    push_final_data(stage)
    return stage


def data(project, args):
    stage = sources(project, args)
    filter_pairs(project, args, stage)
    mix(project, args, stage)


def selftest(project, args):
    stage = make_stage(project, with_encoder=True)
    train_data, val = prepare_training_data(stage)
    run_selftests(stage, stage.skipped_keys, train_data, val)


def evaluate_start(project, args):
    stage = make_stage(project, with_encoder=True)
    _, val = prepare_training_data(stage)
    evaluate_before(stage, val)


def train(project, args):
    stage = make_stage(project, with_encoder=True)
    train_data, val = prepare_training_data(stage)
    training = TrainingRun(stage, train_data, val)
    log("\n" + training.describe())
    training.resume()
    training.train()
    log(training_summary(stage))
    training.export()


def evaluate_end(project, args):
    stage = make_stage(project, with_encoder=True)
    _, val = prepare_training_data(stage)
    evaluate_after(stage, val)


def report(project, args):
    stage = make_stage(project)
    manifest = read_json(stage.work / "final" / "manifest.json")
    estimates = estimate_real_run(stage, manifest) if args.estimates else None
    print(stage_report(stage, manifest, estimates), flush=True)


def all_steps(project, args):
    data(project, args)
    for step in (selftest, evaluate_start, train, evaluate_end, report):
        step(project, args)


def add_arguments(parser):
    parser.add_argument("--estimates", action="store_true", help="report: add the time estimates of the real run")


COMMANDS = {"sources": sources, "filter": filter_pairs, "mix": mix, "data": data, "selftest": selftest,
            "evaluate-before": evaluate_start, "train": train, "evaluate-after": evaluate_end, "report": report,
            "all": all_steps}


def main(argv=None):
    return run("pired contrastive", COMMANDS, argv, add_arguments)


if __name__ == "__main__":
    sys.exit(main())
