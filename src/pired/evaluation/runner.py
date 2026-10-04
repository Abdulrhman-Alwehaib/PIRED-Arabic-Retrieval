import gc
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from ..common.device import device_name, free_gpu
from ..common.io import file_sha256, read_json, write_json, write_text
from ..common.runlog import log
from ..common.text import fmt_hours, md_table
from ..encoder.embedding import load_tokenizer, pooling_mismatches, read_pooling_config
from ..encoder.weights import MODEL_FILES, check_reference_outputs, has_model_files, load_encoder
from ..tokenizer.fingerprint import check_fingerprint
from .mteb_runner import baseline_encoder, mteb, ours_encoder, run_mteb_task, smoke_subset

SAMPLE_ARABIC = ("تقع مدينة الرياض في وسط شبه الجزيرة العربية، وهي عاصمة المملكة العربية السعودية وأكبر مدنها من حيث "
                 "عدد السكان. شهدت المدينة نموا سريعا خلال القرن العشرين، وأصبحت مركزا للتجارة والتعليم والثقافة. "
                 "ويعود تاريخها إلى عصور قديمة، إذ كانت تعرف باسم حجر اليمامة قبل أن تتحول إلى الاسم الحالي. ")

HAND_TABLE = {}


def task_size(task):
    stats = task.metadata.descriptive_stats or {}
    size = {"documents": 0, "queries": 0, "doc_chars": 0, "query_chars": 0}
    for split in task.eval_splits:
        st = stats.get(split, {})
        for subset in task.hf_subsets:
            sub = (st.get("hf_subset_descriptive_stats") or {}).get(subset, st)
            d, q = sub.get("documents_text_statistics") or {}, sub.get("queries_text_statistics") or {}
            if d.get("average_text_length"):
                size["documents"] += round(d["total_text_length"] / d["average_text_length"])
                size["doc_chars"] += d["total_text_length"]
            if q.get("average_text_length"):
                size["queries"] += round(q["total_text_length"] / q["average_text_length"])
                size["query_chars"] += q["total_text_length"]
    return size


def encoded_chars(size, max_tokens, chars_per_token):
    avg = size["doc_chars"] / max(size["documents"], 1)
    return size["documents"] * min(avg, max_tokens * chars_per_token) + size["query_chars"]


def unload_task(task):
    for attr in ("dataset", "corpus", "queries", "relevant_docs"):
        if getattr(task, attr, None) is not None:
            setattr(task, attr, None)
    task.data_loaded = False
    gc.collect()


class Evaluation:
    def __init__(self, project, cfg, device):
        self.project, self.cfg, self.device, self.smoke = project, cfg, device, project.smoke
        self.ours = {}
        self.results = defaultdict(dict)
        self.baseline_label = cfg.models.baseline[0].split("/")[-1]
        self.sanity = {"skipped": "SMOKE"} if self.smoke else {}
        self.mix_scores, self.best_mix, self.small_tasks = {}, None, []

    @property
    def token(self):
        return self.project.hf_token

    def stage3_folder(self):
        m = self.cfg.models
        if not has_model_files(m.stage3_dir):
            repo = self.project.repo_id("contrastive-weak", m.stage3_repo)
            try:
                snapshot_download(repo, allow_patterns=list(MODEL_FILES), local_dir=m.stage3_dir, token=self.token)
            except Exception as e:
                log(f"stage 3: not on disk and not downloadable from {repo} ({type(e).__name__})")
        return Path(m.stage3_dir) if has_model_files(m.stage3_dir) else None

    def stage4_folder(self, name):
        m = self.cfg.models
        local = Path(m.stage4_dir) / name
        if not has_model_files(local) and not self.smoke:
            repo = self.project.repo_id("supervised", m.stage4_repo)
            try:
                snapshot_download(repo, allow_patterns=[f"{name}/{f}" for f in MODEL_FILES], local_dir=m.stage4_dir,
                                  token=self.token)
            except Exception as e:
                log(f"{name}: not on disk and not downloadable from {repo} ({type(e).__name__})")
        return local if has_model_files(local) else None

    def check_our_folder(self, folder):
        m = self.cfg.models
        model = load_encoder(folder).to(self.device).eval()
        diffs = check_reference_outputs(model, folder, atol=1e-4)
        check_fingerprint(load_tokenizer(folder), m.tokenizer_fingerprint, str(folder))
        pool = read_pooling_config(folder)
        bad = pooling_mismatches(pool, m.query_prefix, m.passage_prefix, m.max_query_tokens, m.max_passage_tokens)
        if bad:
            raise AssertionError(f"{folder}: pooling_config.json differs from the config: {bad}")
        del model
        free_gpu()
        return {"folder": str(folder), "sha256": file_sha256(Path(folder) / "model.safetensors"), "step": pool.get("step"),
                "chosen_by": pool.get("chosen_by", pool.get("stage")), "reference_max_diff": max(diffs.values())}

    def find_models(self):
        m = self.cfg.models
        candidates = [("stage 3 (ours)", self.stage3_folder()),
                      ("stage 4 best (ours)", self.stage4_folder(m.stage4_best)),
                      ("stage 4 final (ours)", self.stage4_folder(m.stage4_final)),
                      *[(f"mix {n.rsplit('-', 1)[-1]} (ours)", self.stage4_folder(n)) for n in m.blends]]
        for label, folder in candidates:
            if folder is None:
                hint = (" (made by the blend stage)" if label.startswith("mix")
                        else " (run the supervised stage with --smoke first)" if self.smoke and "4" in label else "")
                log(f"{label}: NOT FOUND, left out{hint}")
                continue
            self.ours[label] = self.check_our_folder(folder)
            o = self.ours[label]
            log(f"{label}: {folder} (step {o['step']}), strict load OK, reference outputs reproduced (max |diff| "
                f"{o['reference_max_diff']:.1e}), tokenizer fingerprint and pooling config OK, weights {o['sha256'][:12]}")
        same = [(a, b) for a in self.ours for b in self.ours if a < b and self.ours[a]["sha256"] == self.ours[b]["sha256"]]
        if same:
            log(f"identical weights: {same} -> evaluated once, the results are copied")
        self.models = [*self.ours, self.baseline_label]
        self.max_doc_tokens = {**{label: m.max_passage_tokens for label in self.ours},
                               self.baseline_label: m.baseline[3]}
        return self.ours

    def make_encoder(self, label):
        m, e = self.cfg.models, self.cfg.eval
        if label in self.ours:
            folder = Path(self.ours[label]["folder"])
            model = load_encoder(folder).to(self.device).eval()
            encoder = ours_encoder(model, load_tokenizer(folder), f"local/{label}", (m.query_prefix, m.passage_prefix),
                                   (m.max_query_tokens, m.max_passage_tokens), e.batch_tokens, self.device,
                                   mask_full_chunks=True, min_batch_tokens=e.min_batch_tokens)
            return encoder, model
        name, q_prefix, p_prefix, max_len = m.baseline
        return baseline_encoder(name, (q_prefix, p_prefix), max_len, e.batch_tokens, self.device,
                                mask_full_chunks=True, min_batch_tokens=e.min_batch_tokens)

    def task_list(self):
        e = self.cfg.eval
        tasks = []
        for t in mteb.get_tasks(languages=["ara"], task_types=["Retrieval"]):
            if t.metadata.name in e.skip or (self.smoke and t.metadata.name != e.smoke_task):
                continue
            subsets = [s for s in (t.hf_subsets or []) if s in e.arabic_subsets]
            if not subsets:
                continue
            t.hf_subsets = subsets
            t.k_values, t._top_k = (1, 3, 5, 10, 20, 100), e.top_k
            tasks.append(t)
        return sorted(tasks, key=lambda t: (t.metadata.name not in e.in_domain, t.metadata.name))

    def plan(self):
        e = self.cfg.eval
        self.tasks = self.task_list()
        sizes = {t.metadata.name: task_size(t) for t in self.tasks}
        rng = random.Random(0)
        speed_texts = []
        for _ in range(e.speed_texts):
            start = rng.randrange(0, 200)
            speed_texts.append((SAMPLE_ARABIC * 20)[start: start + rng.choice([80, 200, 320, 500, 900])])
        speed = {}
        for label in self.models:
            enc, mdl = self.make_encoder(label)
            enc.encode_texts(speed_texts[:20], is_query=False)
            enc.chars, enc.tokens, enc.seconds = 0, 0, 0.0
            enc.encode_texts(speed_texts, is_query=False)
            speed[label] = {"chars_per_s": enc.chars / enc.seconds, "chars_per_token": enc.chars / enc.tokens}
            del enc, mdl
            free_gpu()
        lines = [f"mteb {mteb.__version__}: {len(self.tasks)} Arabic retrieval tasks" + (" (SMOKE: one task)" if self.smoke
                                                                                          else ""),
                 f"{'task':<34}{'kind':<11}{'subsets':<10}{'splits':<17}{'documents':>11}{'queries':>9}  "
                 + "".join(f"{label[:18]:>20}" for label in self.models)]
        total = defaultdict(float)
        for t in self.tasks:
            name, z = t.metadata.name, sizes[t.metadata.name]
            est = {label: encoded_chars(z, self.max_doc_tokens[label], speed[label]["chars_per_token"])
                   / speed[label]["chars_per_s"] for label in self.models}
            for label, s in est.items():
                total[label] += s
            lines.append(f"{name:<34}{'in-domain' if name in e.in_domain else 'zero-shot':<11}{','.join(t.hf_subsets):<10}"
                         f"{','.join(t.eval_splits):<17}{z['documents']:>11,}{z['queries']:>9,}  "
                         + "".join(f"{fmt_hours(est[label]):>20}" for label in self.models))
        lines.append(f"{'total per model':<83}" + "".join(f"{fmt_hours(total[label]):>20}" for label in self.models))
        lines.append(f"speed probe on {self.device}: "
                     + ", ".join(f"{label} {v['chars_per_s']:,.0f} chars/s" for label, v in speed.items()))
        log("\n".join(lines))
        return speed

    def published_results(self):
        out = defaultdict(dict)
        ours = {t.metadata.name: t for t in self.tasks}
        try:
            res = mteb.load_results(models=list(self.cfg.models.published), tasks=list(ours),
                                    download_latest=not self.smoke, validate_and_filter=False, require_model_meta=False)
            for mr in res.model_results:
                for tr in mr.task_results:
                    task = ours.get(tr.task_name)
                    if task is None:
                        continue
                    split = task.eval_splits[-1]
                    rows = [r for r in tr.scores.get(split, []) if r.get("hf_subset") in task.hf_subsets]
                    flags = []
                    if tr.dataset_revision != task.metadata.dataset.get("revision"):
                        flags.append(f"other task data version ({str(tr.dataset_revision)[:8]}, ours "
                                     f"{str(task.metadata.dataset.get('revision'))[:8]})")
                    if not rows:
                        other = [s for s, rs in tr.scores.items() if any(r.get("hf_subset") in task.hf_subsets for r in rs)]
                        if not other:
                            continue
                        split = other[0]
                        rows = [r for r in tr.scores[split] if r.get("hf_subset") in task.hf_subsets]
                        flags.append(f"published split {split}, ours {task.eval_splits[-1]}")
                    entry = {"ndcg@10": rows[0].get("ndcg_at_10"), "recall@100": rows[0].get("recall_at_100"),
                             "split": split, "subset": rows[0].get("hf_subset"), "dataset_revision": tr.dataset_revision,
                             "mteb_version": tr.mteb_version, "revision": mr.model_revision, "flags": flags,
                             "source": "MTEB leaderboard"}
                    old = out[mr.model_name].get(tr.task_name)
                    if old is None or (old["flags"] and not flags):
                        out[mr.model_name][tr.task_name] = entry
        except Exception as e:
            log(f"mteb.load_results failed ({type(e).__name__}: {str(e)[:200]}): only HAND_TABLE is used")
        for (model, task), v in HAND_TABLE.items():
            if v.get("ndcg@10") is not None:
                out[model][task] = {**v, "flags": [f"hand-filled: {v.get('source', '')}"], "source": "hand-filled"}
        self.published = dict(out)
        lines = [f"{'task':<34}" + "".join(f"{m.split('/')[-1]:>28}" for m in self.cfg.models.published)]
        for t in self.tasks:
            cells = []
            for m in self.cfg.models.published:
                r = self.published.get(m, {}).get(t.metadata.name)
                cells.append(f"{r['ndcg@10']:.4f}{' (flagged)' if r['flags'] else ''}"
                             if r and r.get("ndcg@10") is not None else "no Arabic result")
            lines.append(f"{t.metadata.name:<34}" + "".join(f"{c:>28}" for c in cells))
        log("\n".join(lines))
        return self.published

    def eval_repo(self):
        return self.project.repo_id("evaluation")

    def download_results(self):
        if self.smoke or not self.token:
            return
        try:
            snapshot_download(self.eval_repo(), repo_type="dataset", allow_patterns=["results/*"],
                              local_dir=self.cfg.stage_dir, token=self.token)
            log(f"results from {self.eval_repo()}: {len(list(Path(self.cfg.results_dir).rglob('*.json')))} files")
        except Exception as e:
            log(f"no earlier results downloaded from {self.eval_repo()} ({type(e).__name__})")

    def result_path(self, label, task_name):
        return Path(self.cfg.results_dir) / label.replace("/", "_").replace(" ", "_") / f"{task_name}.json"

    def load_result(self, label, task):
        p = self.result_path(label, task.metadata.name)
        if not p.exists():
            return None
        r = read_json(p)
        ok = (r.get("mteb_version") == mteb.__version__
              and r.get("task_revision") == task.metadata.dataset.get("revision")
              and (label not in self.ours or r.get("model_sha256") == self.ours[label]["sha256"]))
        return r if ok else None

    def save_result(self, label, task, scores, source, seconds=0.0):
        r = {"model": label, "task": task.metadata.name, **scores, "mteb_version": mteb.__version__,
             "task_revision": task.metadata.dataset.get("revision"),
             "model_sha256": self.ours[label]["sha256"] if label in self.ours else None, "source": source,
             "seconds": round(seconds, 1), "date": time.strftime("%Y-%m-%d %H:%M"), "smoke": self.smoke}
        write_json(r, self.result_path(label, task.metadata.name))
        return r

    def stage3_result(self, label, task):
        e = self.cfg.eval
        if self.smoke or mteb.__version__ != e.nb04_mteb_version or (task.metadata.name == "MIRACLRetrieval"
                                                                     and label in self.ours):
            return None
        key = {"stage 3 (ours)": "after (this stage)", self.baseline_label: f"baseline ({self.baseline_label})"}.get(label)
        path = Path(e.nb04_results)
        if key is None or not path.exists():
            return None
        r = read_json(path).get(key, {}).get(task.metadata.name)
        if not r:
            return None
        return {"ndcg@10": r["ndcg@10"], "recall@100": r["recall@100"], "subset": r.get("subset"), "split": r.get("split")}

    def load_task(self, task):
        e = self.cfg.eval
        if self.smoke and smoke_subset(task, e.smoke_queries, e.smoke_distractors, self.token):
            return
        task.load_data()

    def evaluate(self, label, task):
        if (r := self.load_result(label, task)) is not None:
            return r
        if (r := self.stage3_result(label, task)) is not None:
            return self.save_result(label, task, r, "stage 3 evaluation (same mteb version)")
        if label in self.ours:
            twin = next((o for o in self.ours if o != label and self.ours[o]["sha256"] == self.ours[label]["sha256"]
                         and self.load_result(o, task) is not None), None)
            if twin is not None:
                src = self.load_result(twin, task)
                return self.save_result(label, task, {k: src[k] for k in ("ndcg@10", "recall@100", "subset", "split")},
                                        f"copied from {twin} (identical weights)")
        if not task.data_loaded:
            self.load_task(task)
        t0 = time.time()
        enc, mdl = self.make_encoder(label)
        try:
            scores = run_mteb_task(enc, task)
        finally:
            del enc, mdl
            free_gpu()
        return self.save_result(label, task, scores, "run here", time.time() - t0)

    def sanity_check(self):
        e = self.cfg.eval
        if self.smoke:
            log(f"SMOKE: the sanity check needs the full MIRACL-ar corpus; skipped (the smoke task is {e.smoke_task})")
            return self.sanity
        if "stage 3 (ours)" not in self.ours or not any(t.metadata.name == "MIRACLRetrieval" for t in self.tasks):
            raise RuntimeError("STOP: the sanity check needs the stage-3 model and the MIRACLRetrieval task")
        miracl = next(t for t in self.tasks if t.metadata.name == "MIRACLRetrieval")
        t0 = time.time()
        r = self.evaluate("stage 3 (ours)", miracl)
        expected, tol = e.stage3_miracl_ndcg10, e.stage3_tolerance
        self.sanity = {"expected": expected, "got": r["ndcg@10"], "tolerance": tol}
        if abs(r["ndcg@10"] - expected) > tol:
            raise RuntimeError(f"STOP: stage 3 gives MIRACL-ar nDCG@10 {r['ndcg@10']:.4f}, not {expected} ± {tol}: the "
                               "loading, prefixes or pooling differ from the stage-3 evaluation. Nothing else ran.")
        log(f"PASS stage 3 reproduces MIRACL-ar nDCG@10: {r['ndcg@10']:.4f} (expected {expected}, allowed ± {tol}; "
            f"{fmt_hours(time.time() - t0)})")
        return self.sanity

    def run_task(self, t, labels):
        name = t.metadata.name
        for label in labels:
            t0 = time.time()
            r = self.evaluate(label, t)
            self.results[label][name] = r
            took = time.time() - t0
            log(f"  {name:<34} {label:<24} nDCG@10 {r['ndcg@10']:.4f}  R@100 {r['recall@100']:.4f}  ({r['source']}"
                + (f", {fmt_hours(took)} now" if took > 2 else "") + ")")
        unload_task(t)

    def run_all(self):
        t_all = time.time()
        mixes = [label for label in self.models if label.startswith("mix ")]
        big = set(self.cfg.eval.big_tasks)
        for t in self.tasks:
            self.run_task(t, [label for label in self.models if label not in mixes or t.metadata.name not in big])
        self.small_tasks = [t.metadata.name for t in self.tasks if t.metadata.name not in big]
        self.mix_scores = {m: float(np.mean([self.results[m][n]["ndcg@10"] for n in self.small_tasks])) for m in mixes}
        self.best_mix = max(self.mix_scores, key=self.mix_scores.get) if self.mix_scores else None
        if self.best_mix:
            log(f"mixes, mean nDCG@10 over the {len(self.small_tasks)} smaller tasks: "
                + ", ".join(f"{m} {s:.4f}" for m, s in sorted(self.mix_scores.items(), key=lambda kv: -kv[1]))
                + f"  ->  {self.best_mix} also runs " + ", ".join(sorted(big & {t.metadata.name for t in self.tasks})))
            fresh = {t.metadata.name: t for t in self.task_list()}
            for name in sorted(big & set(fresh)):
                self.run_task(fresh[name], [self.best_mix])
        log(f"all results in {self.cfg.results_dir} ({fmt_hours(time.time() - t_all)} this session)")

    def table(self, metric):
        in_domain = set(self.cfg.eval.in_domain)
        in_tasks = [t.metadata.name for t in self.tasks if t.metadata.name in in_domain]
        zs_tasks = [t.metadata.name for t in self.tasks if t.metadata.name not in in_domain]
        rows_in = [(label, self.results[label]) for label in self.models] + [
            (f"{m.split('/')[-1]} (from MTEB leaderboard)", self.published.get(m, {}))
            for m in self.cfg.models.published]

        def cell(r):
            if not r or r.get(metric) is None:
                return "-"
            return f"{r[metric]:.4f}" + ("*" if r.get("flags") else "")

        header = ["model", *[f"{n.replace('Retrieval', '')} (in-domain)" for n in in_tasks],
                  *[n.replace("Retrieval", "") for n in zs_tasks], "zero-shot average"]
        rows = []
        for label, res in rows_in:
            zs = [res[n][metric] for n in zs_tasks if n in res and res[n].get(metric) is not None]
            avg = "-"
            if zs:
                avg = f"{np.mean(zs):.4f}" + (f" ({len(zs)} of {len(zs_tasks)} tasks)" if len(zs) < len(zs_tasks) else "")
            rows.append([label, *[cell(res.get(n)) for n in in_tasks + zs_tasks], avg])
        return md_table(header, rows)

    def report(self):
        m = self.cfg.models
        flags = [f"{mod.split('/')[-1]} / {task}: " + "; ".join(r["flags"]) for mod, tasks in self.published.items()
                 for task, r in tasks.items() if r.get("flags")]
        big = sorted(self.cfg.eval.big_tasks)
        mix_note = []
        if self.best_mix:
            mix_note = [f"**Mixes of stage 3 and stage 4 (blend stage):** each ran the {len(self.small_tasks)} smaller "
                        f"tasks; the best by mean nDCG@10 over them, **{self.best_mix}**, also ran " + ", ".join(big)
                        + " (mean over the smaller tasks: " + ", ".join(f"{k} {s:.4f}" for k, s in self.mix_scores.items())
                        + "). The mix was picked on these test tasks, so its scores are test-selected (optimistic); the "
                          "other mixes show `-` on the big tasks."]
        report = "\n\n".join([
            f"# Evaluation{' (SMOKE RUN: one small task, a cut corpus)' if self.smoke else ''}",
            f"`pired.evaluation`, {time.strftime('%Y-%m-%d')}, mteb {mteb.__version__}, on "
            f"{device_name(self.device)}. nDCG@10 and Recall@100 on the monolingual Arabic subsets of every Arabic "
            "retrieval task in mteb (the test split where a task has two).",
            "**Stage 4 trained on the train splits of MIRACL and Mr.TyDi**: those tasks (and MIRACL's hard-negative "
            "versions) are in-domain; the other tasks are zero-shot. These numbers never chose a checkpoint: "
            "stage4-best was chosen by stage 4's validation.",
            "Sanity check: " + (f"stage 3 on MIRACL-ar gives {self.sanity['got']:.4f} (expected "
                                f"{self.sanity['expected']}, allowed ± {self.sanity['tolerance']})."
                                if "got" in self.sanity else "skipped (SMOKE)."),
            *mix_note,
            "## nDCG@10", self.table("ndcg@10"), "## Recall@100", self.table("recall@100"),
            "`-`: no result (the published files of these models often have no Arabic subset). `*`: flagged, see below.",
            "## Flags on published results", "\n".join(f"- {f}" for f in flags) or "none",
            "## Models", md_table(["model", "source", "step / revision"],
                                  [[label, v["folder"], v["step"]] for label, v in self.ours.items()]
                                  + [[self.baseline_label, m.baseline[0], "Hugging Face (latest)"]]
                                  + [[p, "MTEB results repository" if self.published.get(p) else "none found",
                                      ", ".join(sorted({str(r.get("revision"))[:10]
                                                        for r in self.published.get(p, {}).values()}))]
                                     for p in m.published]),
            "## Where each result comes from",
            md_table(["model", "task", "source"],
                     [[label, n, r["source"]] for label, res in self.results.items() for n, r in res.items()]),
        ])
        write_text(report, Path(self.cfg.reports_dir) / "evaluation.md")
        return report

    def upload(self):
        api, root = self.project.api, Path(self.cfg.stage_dir)
        files = sorted(p.relative_to(root).as_posix() for d in (self.cfg.results_dir, self.cfg.reports_dir)
                       for p in Path(d).rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
        repo = self.eval_repo()
        if self.smoke or not self.token:
            log(f"[not uploaded: {'SMOKE' if self.smoke else 'no HF_TOKEN'}] would upload {len(files)} files from {root} "
                f"to the private dataset {repo}")
            return
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(root), allow_patterns=files,
                          commit_message="Evaluation results and report")
        missing = sorted(set(files) - set(api.list_repo_files(repo, repo_type="dataset")))
        if missing:
            raise RuntimeError(f"not on the Hub: {missing[:5]}")
        log(f"uploaded {len(files)} files to {repo}: {sum(f.startswith('results/') for f in files)} results and the "
            "report; all of them are on the Hub")
