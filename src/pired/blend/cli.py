import sys

import torch

from ..common.cli import run
from .blend import Blender, make_config


def build(project, args):
    cfg = make_config(project)
    if args.alpha:
        cfg.alphas = tuple(args.alpha)
    if args.stage4_name:
        cfg.stage4_name = args.stage4_name
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    blender = Blender(project, cfg, device)
    saved = blender.build_all()
    if not args.no_upload:
        blender.upload(saved)


def add_arguments(parser):
    parser.add_argument("--alpha", type=float, action="append", help="share of stage 4 in a mix (repeatable)")
    parser.add_argument("--stage4-name", default=None, help="the stage-4 folder to mix with (default stage4-best)")
    parser.add_argument("--cpu", action="store_true", help="compute the reference outputs on the CPU")
    parser.add_argument("--no-upload", action="store_true", help="build the mixes without uploading them")


COMMANDS = {"build": build}


def main(argv=None):
    return run("pired blend", COMMANDS, argv, add_arguments)


if __name__ == "__main__":
    sys.exit(main())
