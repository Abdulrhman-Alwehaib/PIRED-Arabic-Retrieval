import argparse

from .project import Project


def build_parser(prog, commands, add_arguments=None):
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("command", choices=list(commands))
    parser.add_argument("--smoke", action="store_true", help="tiny sizes, outputs in _smoke_tests/, no uploads")
    parser.add_argument("--root", default=None, help="the project folder (the one with project.json)")
    if add_arguments is not None:
        add_arguments(parser)
    return parser


def run(prog, commands, argv=None, add_arguments=None):
    args = build_parser(prog, commands, add_arguments).parse_args(argv)
    project = Project(args.root, smoke=args.smoke)
    commands[args.command](project, args)
    return 0
