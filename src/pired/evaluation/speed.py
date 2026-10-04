import contextlib
import json
import statistics
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..common.runlog import log
from ..encoder import modeling
from ..encoder.weights import load_encoder


def rotary_native(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos.to(x.dtype) + torch.cat((-x2, x1), dim=-1) * sin.to(x.dtype)


@contextlib.contextmanager
def rotary_in_model_dtype(enabled=True):
    original = modeling.apply_rotary
    if enabled:
        modeling.apply_rotary = rotary_native
    try:
        yield
    finally:
        modeling.apply_rotary = original


def load_texts(texts_dir, sources, per_source):
    texts = {"passages": [], "queries": []}
    for src in sources:
        t = pq.read_table(Path(texts_dir) / src / "part-00000.parquet").slice(0, per_source).to_pydict()
        texts["passages"] += t["passage"]
        texts["queries"] += t["query"]
    return texts


def pooled(model, ids, mask):
    hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return F.normalize((hidden * m).sum(1) / m.sum(1), dim=-1)


class SpeedTest:
    def __init__(self, model_dir, baseline, prefixes, device, batch_size, max_length, repeats):
        self.model_dir, self.baseline, self.device = Path(model_dir), baseline, device
        self.batch_size, self.max_length, self.repeats = batch_size, max_length, repeats
        self.tokenizers = {"mine": AutoTokenizer.from_pretrained(self.model_dir),
                           "e5": AutoTokenizer.from_pretrained(baseline)}
        self.prefixes = prefixes

    def load(self, name, dtype):
        if name == "mine":
            model = load_encoder(self.model_dir)
        else:
            model = AutoModel.from_pretrained(self.baseline, attn_implementation="sdpa")
        return model.to(device=self.device, dtype=dtype).eval()

    def batches(self, name, texts, kind):
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        prefix = self.prefixes[name][kind]
        return [[prefix + texts[j] for j in order[i:i + self.batch_size]] for i in range(0, len(order), self.batch_size)]

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def end_to_end(self, name, model, batches):
        self.sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for b in batches:
                enc = self.tokenizers[name](b, padding=True, truncation=True, max_length=self.max_length,
                                            return_tensors="pt")
                pooled(model, enc["input_ids"].to(self.device, non_blocking=True),
                       enc["attention_mask"].to(self.device, non_blocking=True))
        self.sync()
        return time.perf_counter() - t0

    def model_only(self, model, tensors):
        self.sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for ids, mask in tensors:
                pooled(model, ids, mask)
        self.sync()
        return time.perf_counter() - t0

    def run(self, texts, dtypes):
        results = {"setup": {"device": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU",
                             "batch": self.batch_size, "max_length": self.max_length,
                             "passages": len(texts["passages"]), "queries": len(texts["queries"]),
                             "repeats": self.repeats}}
        for dtype_name, dtype in dtypes:
            models = {n: self.load(n, dtype) for n in ("mine", "e5")}
            results.setdefault("weights_mb", {})[dtype_name] = {
                n: sum(p.numel() * p.element_size() for p in m.parameters()) / 2**20 for n, m in models.items()}
            results.setdefault("params", {n: sum(p.numel() for p in m.parameters()) for n, m in models.items()})
            for kind in ("passages", "queries"):
                bs = {n: self.batches(n, texts[kind], kind) for n in models}
                tensors = {n: [(e["input_ids"].to(self.device), e["attention_mask"].to(self.device)) for e in
                               (self.tokenizers[n](b, padding=True, truncation=True, max_length=self.max_length,
                                                   return_tensors="pt") for b in bs[n])]
                           for n in models}
                tokens = {n: int(sum(int(m.sum()) for _, m in tensors[n])) for n in models}
                peak = {}
                for n, m in models.items():
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        base = torch.cuda.memory_allocated()
                    self.model_only(m, tensors[n])
                    peak[n] = ((torch.cuda.max_memory_allocated() - base) / 2**20 if self.device.type == "cuda"
                               else None)
                e2e, mo = {n: [] for n in models}, {n: [] for n in models}
                for _ in range(self.repeats):
                    for n, m in models.items():
                        e2e[n].append(self.end_to_end(n, m, bs[n]))
                        mo[n].append(self.model_only(m, tensors[n]))
                res = {"tokens": tokens, "token_ratio": tokens["e5"] / tokens["mine"], "activation_peak_mb": peak}
                for label, timing in (("end_to_end", e2e), ("model_only", mo)):
                    med = {n: statistics.median(v) for n, v in timing.items()}
                    res[label] = {"seconds": timing, "median": med,
                                  "texts_per_second": {n: len(texts[kind]) / med[n] for n in med},
                                  "speedup": med["e5"] / med["mine"], "faster_pct": (med["e5"] / med["mine"] - 1) * 100,
                                  "time_saved_pct": (1 - med["mine"] / med["e5"]) * 100}
                    r = res[label]
                    log(f"[{dtype_name}] {kind:8} {label:10}: mine {r['texts_per_second']['mine']:7.0f}/s, e5 "
                        f"{r['texts_per_second']['e5']:7.0f}/s -> {r['speedup']:.2f}x ({r['faster_pct']:.0f}% faster, "
                        f"{r['time_saved_pct']:.0f}% less time)")
                log(f"[{dtype_name}] {kind:8} tokens mine {tokens['mine']:,} e5 {tokens['e5']:,} "
                    f"({res['token_ratio']:.2f}x)")
                results.setdefault(dtype_name, {})[kind] = res
            del models
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return results


def run_speed_test(cfg, model_dir, device, out_file, rotary_fix=False):
    s, m = cfg.speed, cfg.models
    gpu = device.type == "cuda"
    if not gpu:
        torch.set_num_threads(s.cpu_threads)
    texts = load_texts(s.texts_dir, s.sources, s.per_source if gpu else s.cpu_per_source)
    prefixes = {"mine": {"queries": m.query_prefix, "passages": m.passage_prefix},
                "e5": {"queries": m.baseline[1], "passages": m.baseline[2]}}
    test = SpeedTest(model_dir, m.baseline[0], prefixes, device, s.batch_size if gpu else s.cpu_batch_size,
                     s.max_length, s.repeats)
    dtypes = [("fp32", torch.float32)]
    if gpu:
        dtypes = [("bf16", torch.bfloat16)] if rotary_fix else [("bf16", torch.bfloat16), ("fp32", torch.float32)]
    with rotary_in_model_dtype(rotary_fix):
        results = test.run(texts, dtypes)
    results["setup"]["rotary_in_model_dtype"] = rotary_fix
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results
