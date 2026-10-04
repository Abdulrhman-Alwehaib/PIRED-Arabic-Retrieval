import json
import tempfile
from pathlib import Path

import pytest

from pired.tokenizer import checks, training
from pired.tokenizer.config import EVAL_SET_BLOCKLIST, SourceConfig, TokenizerConfig, check_sources
from pired.tokenizer.corpus import is_heldout
from pired.tokenizer.fingerprint import TOKENIZER_FINGERPRINT, vocab_fingerprint
from pired.tokenizer.normalization import build_normalizer, build_pre_tokenizer
from conftest import TOKENIZER_DIR

TEXTS = [
    "تقع مدينة الرياض في وسط شبه الجزيرة العربية، وهي عاصمة المملكة العربية السعودية وأكبر مدنها.",
    "شهدت المدينة نموا سريعا خلال القرن العشرين، وأصبحت مركزا للتجارة والتعليم والثقافة.",
    "إِنَّ اللَّهَ جَمِيلٌ يُحِبُّ الْجَمَالَ، والعدد ١٢٣ والعدد ۴۵۶ في عام 2024.",
    "أعلنت شركة Apple عن هاتف iPhone 15 Pro بسعر 999 دولارا في سبتمبر 2023.",
    "سطر أول\nسطر ثان\r\nسطر ثالث\tبعد تبويب",
] * 40


def test_normalizer_rules():
    normalizer = build_normalizer(TokenizerConfig())
    assert normalizer.normalize_str("إِنَّ اللَّهَ") == "ان الله"
    assert normalizer.normalize_str("الســـــلام") == "السلام"
    assert normalizer.normalize_str("أحمد وإبراهيم وآمنة") == "احمد وابراهيم وامنة"
    assert normalizer.normalize_str("العدد ١٢٣ والعدد ۴۵۶") == "العدد 123 والعدد 456"
    assert normalizer.normalize_str("مستشفى المدينة ومدرسة") == "مستشفى المدينة ومدرسة"
    assert normalizer.normalize_str("  سطر أول  \r\n\r\n   سطر ثان\tمع تبويب  ") == "سطر اول\n سطر ثان مع تبويب"


def test_pre_tokenizer_splits_digits_and_punctuation():
    cfg = TokenizerConfig()
    normalizer, pre = build_normalizer(cfg), build_pre_tokenizer(cfg)
    pieces = [p for p, _ in pre.pre_tokenize_str(normalizer.normalize_str("في 2024!"))]
    assert pieces == ["▁في", "▁2", "0", "2", "4", "!"]


def test_eval_sets_are_refused():
    with pytest.raises(ValueError):
        check_sources([SourceConfig("bad", f"x/{EVAL_SET_BLOCKLIST[0]}-ar", "data/", 1, 1, 1)])


def test_heldout_rule_is_md5_based():
    assert is_heldout("a", 1)
    assert sum(is_heldout(f"text {i}", 50) for i in range(5000)) in range(50, 160)


def test_train_save_and_round_trip():
    cfg = TokenizerConfig(vocab_size=1500)
    raw = training.train_bpe(cfg, [TEXTS[i:i + 50] for i in range(0, len(TEXTS), 50)], show_progress=False)
    backend = training.finish_tokenizer(raw, cfg)
    with tempfile.TemporaryDirectory() as tmp:
        cfg.tokenizer_dir = Path(tmp)
        tok = training.save_tokenizer(backend, cfg, {"test": len(TEXTS)})
        assert tok.convert_ids_to_tokens(list(range(6))) == ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<0x00>"]
        stats = checks.check_round_trip(tok, TEXTS[:20], len(cfg.special_tokens))
        assert stats["unk"] == 0
        checks.check_inference_consistency(tok)
        assert json.loads((Path(tmp) / "tokenizer_training_config.json").read_text())["corpus"] == {"test": 200}


def test_saved_tokenizer_is_the_trained_one():
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("the stage-1 tokenizer is not on this machine")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    assert vocab_fingerprint(tok.get_vocab()) == TOKENIZER_FINGERPRINT
    saved = json.loads((TOKENIZER_DIR / "tokenizer.json").read_text(encoding="utf-8"))
    ours = json.loads(training.empty_tokenizer(TokenizerConfig()).to_str())
    for key in ("normalizer", "pre_tokenizer", "decoder"):
        assert ours[key] == saved[key]
    checks.check_round_trip(tok, TEXTS[:10], 5)
    checks.check_inference_consistency(tok)
