import hashlib
import json

TOKENIZER_FINGERPRINT = "b490ba47ebc5a117"


def vocab_fingerprint(vocab):
    items = sorted(vocab.items(), key=lambda kv: kv[1])
    return hashlib.sha256(json.dumps(items, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def check_fingerprint(tokenizer, expected=TOKENIZER_FINGERPRINT, where=""):
    found = vocab_fingerprint(tokenizer.get_vocab())
    if found != expected:
        raise AssertionError(f"{where} tokenizer fingerprint {found} != {expected}".strip())
    return found
