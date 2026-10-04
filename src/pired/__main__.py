import importlib
import sys

STAGES = ("tokenizer", "pretraining", "contrastive", "supervised", "evaluation", "blend")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in STAGES:
        print(f"usage: python -m pired {{{','.join(STAGES)}}} <command> [--smoke] [--root PATH]")
        return 2
    return importlib.import_module(f"pired.{argv[0]}.cli").main(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
