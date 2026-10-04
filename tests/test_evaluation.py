import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from pired.encoder import modeling
from pired.encoder.embedding import pad_chunks
from pired.evaluation.config import EvaluationConfig
from pired.evaluation.mteb_runner import ours_encoder
from pired.evaluation.runner import Evaluation, encoded_chars, task_size
from pired.evaluation.speed import rotary_in_model_dtype
from conftest import PASSAGE_PREFIX, QUERY_PREFIX, make_text_stage


def test_pad_chunks_groups_by_length():
    ids = [[1] * 5, [1] * 17, [1] * 3, [1] * 9]
    chunks = pad_chunks(ids, chunk_tokens=48, pad_id=0)
    assert [c[0].tolist() for c in chunks] == [[1, 3], [0, 2]]
    assert chunks[0][1].shape == (2, 24) and chunks[0][2] is not None
    full = pad_chunks([[1] * 8, [1] * 8], chunk_tokens=64, pad_id=0)
    assert full[0][2] is None
    assert pad_chunks([[1] * 8, [1] * 8], chunk_tokens=64, pad_id=0, mask_full_chunks=True)[0][2] is not None


def test_retrieval_encoder_matches_the_embedder(tokenizer, device):
    stage, model = make_text_stage(tokenizer, device)
    model.eval()
    texts = ["ما هي عاصمة فرنسا؟", "كتب المؤرخ عن تاريخ المدينة القديمة.", "نعم"]
    encoder = ours_encoder(model, tokenizer, "local/test", (QUERY_PREFIX, PASSAGE_PREFIX), (64, 256), 1024, device)
    got = torch.from_numpy(encoder.encode_texts(texts, is_query=False))
    expected = stage.make_embedder(model).encode(texts, "passage")
    assert (got - expected).abs().max() < 1e-5
    assert encoder.tokens == sum(len(x) for x in stage.embedder.tokenize(texts, "passage"))


def test_task_size_and_estimate():
    stats = {"test": {"hf_subset_descriptive_stats": {"ar": {
        "documents_text_statistics": {"total_text_length": 1000, "average_text_length": 100},
        "queries_text_statistics": {"total_text_length": 50, "average_text_length": 10}}}}}
    task = SimpleNamespace(metadata=SimpleNamespace(descriptive_stats=stats), eval_splits=["test"], hf_subsets=["ar"])
    size = task_size(task)
    assert size == {"documents": 10, "queries": 5, "doc_chars": 1000, "query_chars": 50}
    assert encoded_chars(size, 10, 5.0) == 10 * 50 + 50


def test_results_are_reused_only_when_they_match():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = EvaluationConfig(results_dir=Path(tmp))
        project = SimpleNamespace(smoke=True, hf_token=None)
        evaluation = Evaluation(project, cfg, torch.device("cpu"))
        evaluation.ours = {"stage 3 (ours)": {"sha256": "abc"}}
        task = SimpleNamespace(metadata=SimpleNamespace(name="T", dataset={"revision": "r1"}))
        evaluation.save_result("stage 3 (ours)", task, {"ndcg@10": 0.5, "recall@100": 0.9}, "run here")
        assert evaluation.load_result("stage 3 (ours)", task)["ndcg@10"] == 0.5
        evaluation.ours["stage 3 (ours)"]["sha256"] = "other"
        assert evaluation.load_result("stage 3 (ours)", task) is None


def test_rotary_switch_is_restored():
    original = modeling.apply_rotary
    with rotary_in_model_dtype(True):
        assert modeling.apply_rotary is not original
        x = torch.randn(1, 2, 4, 8)
        cos, sin = modeling.rotary_cos_sin(4, 8, 10_000.0, x.device)
        assert torch.allclose(modeling.apply_rotary(x, cos, sin), original(x, cos, sin), atol=1e-6)
    assert modeling.apply_rotary is original
