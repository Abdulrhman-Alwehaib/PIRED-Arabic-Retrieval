import sys
from pathlib import Path

import torch

from ..common.cli import run
from ..common.device import setup_device
from ..common.runlog import log
from .config import make_config
from .runner import Evaluation
from .speed import run_speed_test


def make_evaluation(project):
    cfg = make_config(project)
    device = setup_device()
    if device.type != "cuda" and not project.smoke:
        raise RuntimeError("STOP: no CUDA GPU found; only --smoke runs on the CPU")
    return Evaluation(project, cfg, device)


def models(project, args):
    make_evaluation(project).find_models()


def plan(project, args):
    evaluation = make_evaluation(project)
    evaluation.find_models()
    evaluation.plan()


def evaluate(project, args):
    evaluation = make_evaluation(project)
    evaluation.download_results()
    evaluation.find_models()
    evaluation.plan()
    evaluation.published_results()
    evaluation.sanity_check()
    evaluation.run_all()
    print(evaluation.report(), flush=True)
    evaluation.upload()


def speed(project, args):
    cfg = make_config(project)
    device = torch.device("cpu") if args.cpu else setup_device()
    model_dir = Path(args.model) if args.model else Path(cfg.models.stage4_dir) / "stage4-blend-0.5"
    name = "speed_results" + ("_gpu" if device.type == "cuda" else "") + ("_fix" if args.rotary_fix else "") + ".json"
    out = Path(cfg.stage_dir) / "speed_test" / name
    run_speed_test(cfg, model_dir, device, out, rotary_fix=args.rotary_fix)
    log(f"results -> {out}")


def add_arguments(parser):
    parser.add_argument("--model", default=None, help="speed: the model folder (default stage4-blend-0.5)")
    parser.add_argument("--cpu", action="store_true", help="speed: run on the CPU")
    parser.add_argument("--rotary-fix", action="store_true", help="speed: rotary math in the model dtype (bf16 only)")


COMMANDS = {"models": models, "plan": plan, "run": evaluate, "speed": speed}


def main(argv=None):
    return run("pired evaluation", COMMANDS, argv, add_arguments)


if __name__ == "__main__":
    sys.exit(main())
