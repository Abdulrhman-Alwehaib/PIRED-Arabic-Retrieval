import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from pired.common.device import exact_fp32
from pired.supervised.checks import Part1State, broken_translation, check_dedupe, holdout
from pired.supervised.config import FilterSettings, GenSettings, SourceSettings
from pired.supervised.generation import banned_regex, copied_run, parse_output
from pired.supervised.mining import cap_counts, cap_synthetic, sigmoid
from pired.supervised.training import (Batch, BatchLoader, contrastive_loss, full_batch_step,
                                                grad_cache_step, retrieval_metrics)
from conftest import make_text_stage

QUERIES = ["ما هي عاصمة فرنسا؟", "كيف أحضر القهوة العربية؟", "متى تأسست المدينة؟", "من كتب الرواية؟"]
PASSAGES = ["باريس هي عاصمة فرنسا وأكبر مدنها.", "تحضر القهوة العربية بغلي البن مع الهيل.",
            "تأسست المدينة في القرن الثامن.", "كتب الرواية أديب معروف.", "نص آخر عن الطقس اليوم.",
            "خبر رياضي عن مباراة كرة القدم.", "وصفة طعام سهلة وسريعة."]


def fake_rows(source, idx):
    q = [QUERIES[int(i) % 4] for i in idx]
    pos = [[PASSAGES[int(i) % 4], PASSAGES[(int(i) + 1) % 4]] for i in idx]
    neg = [[PASSAGES[4 + (int(i) + k) % 3] for k in range(2)] for i in idx]
    scores = [[0.9, 0.1] for _ in idx]
    return q, pos, neg, scores


def test_hard_negative_loss_masks_positives_and_skips():
    q = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    p = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    p_hash = torch.tensor([10, 11, 12, 20, 10, 22])
    pos_hash = torch.tensor([[10, 99], [20, -1]])
    batch = Batch("a", 2, [], [], torch.tensor([1, 2]), p_hash, pos_hash, 0, torch.tensor([[0, 1]]))
    loss, stats = contrastive_loss(q, p, batch, 0.02, bidirectional=False)
    assert int(stats["masked"]) == 2 and stats["skipped_negs"] == 1 and torch.isfinite(loss)
    _, stats_b = contrastive_loss(q, p, batch, 0.02, bidirectional=True)
    assert "loss_pq" in stats_b


def test_grad_cache_equals_full_batch(tokenizer, device):
    stage, model = make_text_stage(tokenizer, device, temperature=0.02, bidirectional=False, max_neg_score=0.75)
    data = SimpleNamespace(sizes={"a": 8}, rows=fake_rows)
    loader = BatchLoader(stage, data, 2, 4, seed=0, chunk_tokens=64)
    batch = loader.rows_batch("a", np.arange(4), chunk_tokens=64)
    assert batch.skip.tolist() == [[i, 3 * i + 1] for i in range(4)]
    assert batch.pos_hash.shape == (4, 2)

    def grads(step_fn):
        model.train()
        model.zero_grad(set_to_none=True)
        loss, _ = step_fn(model, batch, stage.cfg.train, device, False)
        return loss.item(), torch.cat([p.grad.flatten() for p in model.parameters()])

    with exact_fp32():
        l_full, g_full = grads(full_batch_step)
        l_gc, g_gc = grads(grad_cache_step)
    assert abs(l_full - l_gc) < 1e-5 and ((g_gc - g_full).norm() / g_full.norm()).item() < 1e-4


def test_retrieval_metrics():
    top = np.array([[5, 1, 2], [7, 8, 9]])
    m = retrieval_metrics(top, [{5}, {9}])
    assert m["recall@10"] == 1.0 and abs(m["ndcg@10"] - (1 + 1 / np.log2(4)) / 2) < 1e-9


def test_cap_synthetic_keeps_two_queries_per_passage():
    q = pd.DataFrame({"group": ["p1"] * 3 + ["p2"] * 3 + ["p3"],
                      "qtype": ["question", "search_query", "paraphrase"] * 2 + ["question"]})
    capped = cap_synthetic(q, cap=10, per_passage=2, seed=0)
    assert capped.groupby("group").size().to_dict() == {"p1": 2, "p2": 2, "p3": 1}
    assert set(capped[capped["group"] == "p1"]["qtype"]) >= {"search_query"}
    assert len(cap_synthetic(q, cap=4, per_passage=2, seed=0)) == 4


def test_mix_caps():
    sources = {"mmarco": SourceSettings("mmarco", "", "mmarco", False, None, None, ""),
               "synthetic": SourceSettings("synthetic", "", "wiki", False, None, None, ""),
               "miracl": SourceSettings("miracl", "", "wiki", True, None, None, "")}
    stage = SimpleNamespace(smoke=False, sources=sources, cfg=SimpleNamespace(filters=FilterSettings()))
    n = cap_counts(stage, {"mmarco": 1000, "synthetic": 1000, "miracl": 100})
    total = sum(n.values())
    assert n["mmarco"] <= 0.40 * total + 1 and n["synthetic"] <= 0.65 * total + 1
    assert abs(sigmoid(0.0) - 0.5) < 1e-12


def test_holdout_keeps_groups_together():
    q = pd.DataFrame({"group": [f"g{i // 3}" for i in range(300)]})
    flags = holdout(q, 30, seed=1)
    assert 30 <= flags.sum() <= 32
    assert (pd.Series(flags).groupby(q["group"]).nunique() == 1).all()


def test_translation_and_generation_checks():
    f = FilterSettings()
    assert broken_translation(f, "this is a fully english sentence", True).startswith("untranslated")
    assert broken_translation(f, "كلمة كلمة كلمة كلمة اخرى", True).startswith("broken")
    assert broken_translation(f, "سؤال جيد عن المدينة", True) is None
    queries, why, leak = parse_output('<think>x</think>```json\n{"question": "ما؟", "search_query": "a b", '
                                      '"paraphrase": "هل؟"}\n```')
    assert queries == {"question": "ما؟", "search_query": "a b", "paraphrase": "هل؟"} and leak
    assert parse_output("no json")[0] is None


def test_banned_words_and_copying(tokenizer):
    stage = SimpleNamespace(normalize=tokenizer.backend_tokenizer.normalizer.normalize_str,
                            cfg=SimpleNamespace(gen=GenSettings()))
    banned = banned_regex(stage)
    for text, removed in (("ما الذي يذكره النص عن النيل؟", True), ("بحسب الفقرة، متى تأسست المدينة؟", True),
                          ("من كتب المقالة؟", True), ("ما سبب النصر في المعركة؟", False)):
        assert bool(banned.search(stage.normalize(text))) == removed
    passage = "تقع مدينة الرياض في وسط شبه الجزيرة العربية وهي عاصمة المملكة"
    assert copied_run(stage, "اين تقع مدينة الرياض في وسط البلاد", passage, 5) == "تقع مدينة الرياض في وسط"
    assert copied_run(stage, "ما عاصمة المملكة", passage, 5) is None


def test_dedupe_merges_positives(tokenizer):
    with tempfile.TemporaryDirectory() as tmp:
        sources = {s: SourceSettings(s, "", "wiki", True, None, None, "") for s in ("a", "b")}
        stage = SimpleNamespace(source_names=["a", "b"], steps1=Path(tmp), sources=sources,
                                normalize=tokenizer.backend_tokenizer.normalizer.normalize_str)
        columns = dict(domain="wiki", group="", qtype="", is_human=True)
        queries = {"a": pd.DataFrame([{"qid": "a:1", "source": "a", "query": "ما هي عاصمة قطر؟", "pos_pids": ["p1"],
                                       "neg_pids": [], **columns}]),
                   "b": pd.DataFrame([{"qid": "b:1", "source": "b", "query": "ما هي عاصمة قطر", "pos_pids": ["p2"],
                                       "neg_pids": ["p3"], **columns}])}
        passages = pd.DataFrame({"pid": ["p1", "p2", "p3"], "text": ["الدوحة", "الدوحة عاصمة", "الرياض"],
                                 "domain": "wiki", "orig_id": ""}).set_index("pid")
        state = Part1State(stage, queries, passages)
        stats = state.run_check("dedupe", check_dedupe, {"version": 1})
        assert stats["out"] == {"a": 1, "b": 0}
        assert list(state.queries["a"]["pos_pids"][0]) == ["p1", "p2"]
        assert list(state.queries["a"]["neg_pids"][0]) == ["p3"]
        again = Part1State(stage, queries, passages)
        assert again.run_check("dedupe", check_dedupe, {"version": 1})["out"] == stats["out"]
