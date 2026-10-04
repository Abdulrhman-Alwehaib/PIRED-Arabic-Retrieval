import contextlib
import math
from types import SimpleNamespace

import numpy as np
import torch

from pired.common.device import exact_fp32
from pired.contrastive import context
from pired.contrastive.config import DataSettings, FilterSettings, SourceSettings
from pired.contrastive.dedupe import MinHasher, near_duplicate_keepers
from pired.contrastive.mix import capped_counts
from pired.contrastive.sources import answer_window, mined_questions, span_pair
from pired.contrastive.steps import (WORD_RE, arabic_share, clean_text, decide_language, decide_length,
                                              decide_near_identical, fit_to_length, query_key, word_jaccard)
from pired.contrastive.training import (BatchLoader, contrastive_loss, full_batch_step, grad_cache_step,
                                                 make_batch)
from conftest import make_text_stage

QUERIES = ["ما هي عاصمة فرنسا؟", "كيف أحضر القهوة العربية؟", "متى تأسست المدينة؟", "من كتب الرواية؟"] * 4
PASSAGES = ["باريس هي عاصمة فرنسا وأكبر مدنها وتقع على نهر السين.",
            "تحضر القهوة العربية بغلي البن مع الهيل وتقدم في فناجين صغيرة.",
            "تأسست المدينة في القرن الثامن على يد الخليفة وأصبحت مركزا للعلم.",
            "كتب الرواية أديب معروف ونشرت في القاهرة قبل خمسين عاما."] * 4


def activate(stage):
    context.ACTIVE["stage"] = stage
    return stage


def test_loss_masks_duplicate_texts():
    q = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    p = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    q_hash = torch.tensor([1, 2, 3, 4])
    p_hash = torch.tensor([10, 10, 11, 1])
    loss, stats = contrastive_loss(q, p, q_hash, p_hash, 0.02, bidirectional=False)
    assert int(stats["masked"]) == 3 and torch.isfinite(loss)
    loss_b, stats_b = contrastive_loss(q, p, q_hash, p_hash, 0.02, bidirectional=True)
    assert "loss_pq" in stats_b and torch.isfinite(loss_b)


def test_grad_cache_equals_full_batch(tokenizer, device):
    stage, model = make_text_stage(tokenizer, device, temperature=0.02, bidirectional=True)
    batch = make_batch(stage, QUERIES, PASSAGES, "a", chunk_tokens=64)
    assert len(batch.q_chunks) > 1 and len(batch.p_chunks) > 1
    s = stage.cfg.train

    def grads(step_fn):
        model.train()
        model.zero_grad(set_to_none=True)
        loss, _ = step_fn(model, batch, s, device, False)
        return loss.item(), torch.cat([p.grad.flatten() for p in model.parameters()])

    with exact_fp32():
        l_full, g_full = grads(full_batch_step)
        l_gc, g_gc = grads(grad_cache_step)
    assert abs(l_full - l_gc) < 1e-5
    assert ((g_gc - g_full).norm() / g_full.norm()).item() < 1e-4
    assert l_full <= math.log(len(QUERIES)) + 1.0


def test_batch_loader_is_deterministic(tokenizer, device):
    stage, _ = make_text_stage(tokenizer, device)
    data = SimpleNamespace(sizes={"a": 10, "b": 6},
                           rows=lambda source, idx: ([QUERIES[int(i) % 4] for i in idx],
                                                     [PASSAGES[int(i) % 4] for i in idx]))
    a, b = BatchLoader(stage, data, 4, seed=1, chunk_tokens=128), BatchLoader(stage, data, 4, seed=1, chunk_tokens=128)
    assert a.schedule == b.schedule and a.batches_per_source() == {"a": 2, "b": 1}
    assert all(np.array_equal(a.perms[s], b.perms[s]) for s in data.sizes)
    assert torch.equal(a.get(0).q_hash, b.get(0).q_hash)


def test_minhash_estimates_jaccard():
    base = "ذهب الوفد الرسمي إلى العاصمة لبحث التعاون الاقتصادي بين البلدين في مجالات الطاقة والنقل والتجارة " * 3
    variants = [base, base.replace("الطاقة", "الزراعة"), "نص مختلف تماما عن الاقتصاد يتحدث عن كرة القدم والملاعب"]
    hasher = MinHasher(1234, 64, 5)
    sig = hasher.signatures(variants)
    grams = [set(zip(*[WORD_RE.findall(v)[k:] for k in range(5)])) for v in variants]
    exact = len(grams[0] & grams[1]) / len(grams[0] | grams[1])
    assert abs(float((sig[0] == sig[1]).mean()) - exact) < 0.2
    assert float((sig[0] == sig[2]).mean()) < 0.1
    keeper, _ = near_duplicate_keepers(np.concatenate([sig, sig[:1]]), 16, 0.8)
    assert keeper.tolist() == [0, 1, 2, 0]


def test_clean_text_and_length_cut(tokenizer, device):
    text, rules = clean_text("<p>زر https://example.com الآن</p> أو راسل a@b.com   فورا")
    assert rules == ["html", "url", "email", "whitespace"] and "http" not in text and "@" not in text
    stage, _ = make_text_stage(tokenizer, device)
    long_text = "كلمة طويلة جدا " * 200
    out, counts, cut = fit_to_length(stage, [long_text, "قصير"], stage.p_prefix, stage.n_p_prefix, 32)
    assert cut == [True, False]
    assert len(tokenizer(stage.p_prefix + out[0])["input_ids"]) <= 32
    assert long_text.startswith(out[0]) and long_text[len(out[0])] == " "
    assert counts[0] == len(tokenizer(out[0], add_special_tokens=False)["input_ids"])


def test_row_filters():
    activate(SimpleNamespace(cfg=SimpleNamespace(filters=FilterSettings())))
    reasons, _ = decide_length({"q_tokens": [2, 5, 5], "p_tokens": [30, 10, 30]})
    assert reasons == ["query < 3 tokens", "passage < 20 tokens", None]
    reasons, _ = decide_language({"q_norm": ["hello world", "مرحبا"], "p_norm": ["نص عربي", "english text here"]})
    assert reasons[0].startswith("query") and reasons[1].startswith("passage")
    reasons, _ = decide_near_identical({"q_norm": ["نفس النص تماما", "سؤال"], "p_norm": ["نفس النص تماما", "جواب اخر"]})
    assert reasons[0] is not None and reasons[1] is None
    assert query_key("Hello، World!") == "hello world" and word_jaccard("a b", "a b") == 1.0
    assert arabic_share("abc عربي") == 4 / 7


def test_answer_window_keeps_the_answer():
    context_text = "مقدمة.\n" + "جملة طويلة عن موضوع آخر. " * 40 + "الجواب هنا. " + "تكملة النص. " * 40
    start = context_text.find("الجواب")
    window = answer_window(context_text, start, start + 6, 300)
    assert "الجواب" in window and len(window) <= 300


def test_pair_makers(tokenizer, device):
    stage, _ = make_text_stage(tokenizer, device)
    stage.cfg = SimpleNamespace(data=DataSettings(), train=stage.cfg.train)
    activate(stage)
    text = ("مقدمة المقال.\nما هي فوائد الشاي الأخضر؟\nيحتوي الشاي الأخضر على مضادات أكسدة كثيرة تساعد الجسم على "
            "مقاومة الالتهابات، كما يحسن التركيز ويساعد على حرق الدهون عند شربه باعتدال يوميا.\nهل له أضرار؟\nنعم.")
    assert mined_questions(text) == [("ما هي فوائد الشاي الأخضر؟", text.split("\n")[2]), ("هل له أضرار؟", "نعم.")]
    enc = tokenizer("كلمة " * 300, add_special_tokens=False, return_offsets_mapping=True)
    q, p = span_pair("كلمة " * 300, enc["input_ids"], enc["offset_mapping"], np.random.default_rng(0))
    assert 16 <= len(q.split()) <= 64 and 64 <= len(p.split()) <= 256


def test_source_caps_respect_the_share():
    sources = {"big": SourceSettings("big", "", 10_000, None, ""), "small": SourceSettings("small", "", None, None, "")}
    stage = SimpleNamespace(smoke=False, sources=sources, cfg=SimpleNamespace(filters=FilterSettings()))
    counts = capped_counts(stage, {"big": 9_000, "small": 1_000})
    assert counts["big"] <= 0.4 / 0.6 * counts["small"] + 1
    with contextlib.suppress(KeyError):
        context.ACTIVE.pop("stage")
