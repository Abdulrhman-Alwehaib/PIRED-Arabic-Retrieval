import sys
import time

from transformers import AutoTokenizer

from ..common.cli import run
from ..common.runlog import log
from .checks import (MIXED_EXAMPLES, check_inference_consistency, check_round_trip, fertility_report, format_fertility,
                     show_tokens, vocab_composition)
from .config import make_config
from .corpus import build_corpus, load_heldout
from .fingerprint import vocab_fingerprint
from .training import finish_tokenizer, save_tokenizer, train_bpe_on_corpus


def corpus(project, args):
    cfg = make_config(project)
    t0 = time.time()
    manifest = build_corpus(cfg, project.hf_token)
    for name, c in manifest["counts"].items():
        log(f"{name:6} train {c['train_docs']:>9,} docs {c['train_words']:>13,} words | "
            f"held-out {c['heldout_docs']:>6,} docs {c['heldout_words']:>11,} words")
    total = sum(c["train_words"] for c in manifest["counts"].values())
    log(f"total training words: {total:,} ({time.time() - t0:.0f}s)")
    return manifest


def train(project, args):
    cfg = make_config(project)
    manifest = build_corpus(cfg, project.hf_token)
    t0 = time.time()
    raw = train_bpe_on_corpus(cfg, manifest)
    log(f"trained in {time.time() - t0:.0f}s, vocab before byte fallback: {raw.get_vocab_size()}")
    tok = save_tokenizer(finish_tokenizer(raw, cfg), cfg, manifest["counts"])
    log(f"saved {type(tok).__name__} with {len(tok):,} tokens to {cfg.tokenizer_dir}, fingerprint "
        f"{vocab_fingerprint(tok.get_vocab())}")


def check(project, args):
    cfg = make_config(project)
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_dir)
    heldout = load_heldout(cfg)
    stats = check_round_trip(tok, [d["text"] for d in heldout], len(cfg.special_tokens))
    log(f"round trip: {stats['texts']} texts OK, [UNK] {stats['unk']}, byte-fallback tokens "
        f"{stats['byte_fallback_share']:.4%}")
    check_inference_consistency(tok)
    log("normalization at inference and word-initial consistency: OK")
    for text in MIXED_EXAMPLES:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        log(f"{len(text.split()):>2} words -> {len(ids):>2} tokens | {show_tokens(tok, ids)}")
    log(f"vocabulary: {dict(vocab_composition(tok, cfg.special_tokens).most_common())}")
    log(f"fingerprint {vocab_fingerprint(tok.get_vocab())}")


def fertility(project, args):
    cfg = make_config(project)
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_dir)
    reference = AutoTokenizer.from_pretrained(cfg.reference_tokenizer)
    report = fertility_report(tok, reference, load_heldout(cfg), cfg)
    print(format_fertility(report), flush=True)


def push(project, args):
    cfg = make_config(project)
    if not cfg.push_to_hub or not project.hf_token:
        log("push skipped (smoke run, push_to_hub off or no HF_TOKEN)")
        return
    repo_id = project.repo_id("tokenizer")
    project.api.create_repo(repo_id, private=True, exist_ok=True)
    project.api.upload_folder(repo_id=repo_id, folder_path=str(cfg.tokenizer_dir), commit_message="Upload tokenizer")
    log(f"pushed to {repo_id}")


def all_steps(project, args):
    for step in (corpus, train, check, fertility, push):
        step(project, args)


COMMANDS = {"corpus": corpus, "train": train, "check": check, "fertility": fertility, "push": push, "all": all_steps}


def main(argv=None):
    return run("pired tokenizer", COMMANDS, argv)


if __name__ == "__main__":
    sys.exit(main())
