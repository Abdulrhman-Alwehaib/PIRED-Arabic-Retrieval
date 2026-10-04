import json
import random
import re
from collections import Counter
from pathlib import Path

from .normalization import METASPACE
from .training import N_BYTE_TOKENS

EDGE_CASES = [
    "مرحبا",
    "  نص مع مسافات   في البداية والنهاية  ",
    "إِنَّ اللّٰهَ جَمِيلٌ يُحِبُّ الْجَمَالَ",
    "سطر أول\n\n\nسطر ثان\r\nسطر ثالث\tبعد تبويب",
    "العدد ١٢٣٤٥ والعدد ۶۷۸۹ و 3.14 و 1,000,000",
    "emoji 😀🎉 وصينية 中文字符 وكورية 한국어 وعبرية עברית",
    f"رموز نادرة: ∮ ⅋ ℵ ☃ {chr(0x1D11E)} و {METASPACE} (U+2581) حرفيا",
    "code: def f(x): return x**2  # تعليق",
    "URL https://example.com/path?q=كلمة&x=1 و email@test.org",
    "Mixed English sentence with Arabic كلمات in the middle.",
    f"{chr(0x0627)}{chr(0x0654)} decomposed hamza و {chr(0xFEF7)} presentation form",
    "",
]

MIXED_EXAMPLES = [
    "أعلنت شركة Apple عن هاتف iPhone 15 Pro بسعر 999 دولارا في سبتمبر 2023.",
    "تواصل معنا عبر info@example.com أو زر https://www.example.com/ar/news?id=42",
    "درجة الحرارة ٣٥ درجة مئوية، والرطوبة ٦٠٪ يوم 12/08/2024",
    "فيروس COVID-19 وسلالة SARS-CoV-2 وجائحة عام 2020",
    "The Transformer architecture (Vaswani et al., 2017) غيّر معالجة اللغة الطبيعية.",
    "سعر السهم ارتفع بنسبة 3.75% ليصل إلى 1,250.50 ريال",
]

ARABIC_RANGE = f"{chr(0x0600)}-{chr(0x06FF)}{chr(0x0750)}-{chr(0x077F)}"
LATIN_RANGE = f"A-Za-z{chr(0x00C0)}-{chr(0x024F)}"
DIGIT_RANGE = f"0-9{chr(0x0660)}-{chr(0x0669)}{chr(0x06F0)}-{chr(0x06F9)}"
ARABIC_WORD = re.compile(f"^[{ARABIC_RANGE}]+$")
LATIN_WORD = re.compile(f"^[{LATIN_RANGE}]+$")
NUMBER_WORD = re.compile(f"^[{DIGIT_RANGE}]+([.,][{DIGIT_RANGE}]+)*$")
EDGE_PUNCT = re.compile(r"^[\W_]+|[\W_]+$")


def round_trip_failures(tok, texts):
    normalizer = tok.backend_tokenizer.normalizer
    unk_id = tok.unk_token_id
    failures = []
    encoded = tok(texts, add_special_tokens=False, verbose=False)["input_ids"]
    for text, ids in zip(texts, encoded):
        normalized = normalizer.normalize_str(text)
        decoded = tok.decode(ids)
        with_specials = tok.decode(tok(text, verbose=False)["input_ids"], skip_special_tokens=True)
        renormalized_ids = tok(normalized, add_special_tokens=False, verbose=False)["input_ids"]
        if decoded != normalized or with_specials != normalized or renormalized_ids != ids or unk_id in ids:
            failures.append((text[:80], normalized[:80], decoded[:80]))
    return failures


def check_round_trip(tok, texts, n_special):
    texts = list(texts) + EDGE_CASES
    failures = round_trip_failures(tok, texts)
    if failures:
        raise AssertionError(f"round trip failed for {len(failures)} of {len(texts)} texts, e.g. {failures[:3]}")
    all_ids = [i for ids in tok(texts, add_special_tokens=False, verbose=False)["input_ids"] for i in ids]
    byte_ids = set(range(n_special, n_special + N_BYTE_TOKENS))
    return {"texts": len(texts), "unk": all_ids.count(tok.unk_token_id),
            "byte_fallback_share": sum(i in byte_ids for i in all_ids) / max(len(all_ids), 1)}


def check_inference_consistency(tok):
    if tok("إِنَّ الْعِلْمَ نُورٌ")["input_ids"] != tok("ان العلم نور")["input_ids"]:
        raise AssertionError("diacritized and plain text give different ids")
    first_word = tok("العلم", add_special_tokens=False)["input_ids"]
    if tok(" \n  العلم", add_special_tokens=False)["input_ids"] != first_word:
        raise AssertionError("a word after stripped whitespace tokenizes differently")
    if tok("طلب العلم", add_special_tokens=False)["input_ids"][-len(first_word):] != first_word:
        raise AssertionError("a word mid-sentence tokenizes differently")


def show_tokens(tok, ids):
    return " ".join(t.replace("\n", "⏎") for t in tok.convert_ids_to_tokens(ids))


def categorized_words(texts, per_category, seed):
    buckets = {"arabic": [], "latin": [], "number": []}
    for text in texts:
        for raw in text.split():
            word = EDGE_PUNCT.sub("", raw)
            if ARABIC_WORD.match(word):
                buckets["arabic"].append(word)
            elif LATIN_WORD.match(word):
                buckets["latin"].append(word)
            elif NUMBER_WORD.match(word):
                buckets["number"].append(word)
    rng = random.Random(seed)
    return {k: rng.sample(v, min(per_category, len(v))) for k, v in buckets.items()}


def fertility_stats(tok, texts_by_source, words):
    stats = {"vocab_size": len(tok)}
    total_tokens = total_words = 0
    for name, texts in texts_by_source.items():
        n_tokens = sum(len(ids) for ids in tok(texts, add_special_tokens=False, verbose=False)["input_ids"])
        n_words = sum(len(t.split()) for t in texts)
        stats[f"docs_{name}"] = n_tokens / n_words
        total_tokens, total_words = total_tokens + n_tokens, total_words + n_words
    stats["docs_all"] = total_tokens / total_words
    unk = tok.unk_token_id
    for category, sample in words.items():
        encoded = tok(sample, add_special_tokens=False)["input_ids"]
        stats[f"{category}_fertility"] = sum(map(len, encoded)) / max(len(sample), 1)
        stats[f"{category}_single_token"] = sum(len(ids) == 1 for ids in encoded) / max(len(sample), 1)
        stats[f"{category}_unk_rate"] = sum(ids.count(unk) for ids in encoded) / max(sum(map(len, encoded)), 1)
    return stats


def fertility_report(tok, reference, heldout_docs, cfg):
    texts_by_source = {}
    for doc in heldout_docs:
        texts_by_source.setdefault(doc["source"], []).append(doc["text"])
    words = categorized_words([d["text"] for d in heldout_docs], cfg.fertility_words_per_category, cfg.seed)
    report = {"ours": fertility_stats(tok, texts_by_source, words),
              "e5": fertility_stats(reference, texts_by_source, words)}
    (Path(cfg.tokenizer_dir) / "fertility_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def format_fertility(report):
    lines = [f"{'metric':<24}{'ours':>12}{'e5-base':>12}",
             f"{'vocab_size':<24}{report['ours']['vocab_size']:>12,}{report['e5']['vocab_size']:>12,}"]
    for key in (k for k in report["ours"] if k != "vocab_size"):
        percent = "single_token" in key or "unk_rate" in key
        cells = [f"{report[side][key]:>12.2%}" if percent else f"{report[side][key]:>12.3f}" for side in ("ours", "e5")]
        lines.append(f"{key:<24}{cells[0]}{cells[1]}")
    return "\n".join(lines)


def token_script(token, special_tokens):
    body = token.replace(METASPACE, "")
    if token in special_tokens:
        return "special"
    if re.fullmatch(r"<0x[0-9A-F]{2}>", token):
        return "byte"
    if not body:
        return "space"
    if re.search(f"[{ARABIC_RANGE}]", body):
        return "arabic"
    if re.search(f"[{LATIN_RANGE}]", body):
        return "latin"
    if body.isdigit():
        return "digit"
    return "punct/other"


def vocab_composition(tok, special_tokens):
    return Counter(token_script(t, special_tokens) for t in tok.get_vocab())
