import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import regex
import requests
import torch

from ..common.device import free_gpu
from ..common.io import write_parquet
from ..common.runlog import log
from ..common.text import fmt_hours, short
from ..contrastive.steps import WORD_RE, arabic_share
from .queries import SourceBuilder

QUERY_KEYS = ("question", "search_query", "paraphrase")
JSON_SCHEMA = {"type": "object", "properties": {k: {"type": "string"} for k in QUERY_KEYS},
               "required": list(QUERY_KEYS), "additionalProperties": False}
SYSTEM_PROMPT = "You write Arabic search queries for training a search engine. Reply with one JSON object only."
USER_PROMPT = """Passage (Arabic):
\"\"\"
{passage}
\"\"\"

Write three different queries in Arabic that this passage answers:
1. "question": a natural, complete question in Modern Standard Arabic. Start it with «{qword}» if that fits the passage; otherwise start it with another question word.
2. "search_query": a short query of 2 to 6 words, as typed into a search box, not a full sentence (for example: عاصمة قطر).
3. "paraphrase": a question about the passage that uses different words than the passage (synonyms, another sentence structure).

Rules:
- Each query must be answerable from this passage alone.
- Do not copy long phrases from the passage.
- Never mention the passage itself: no words such as النص، الفقرة، المقال.

Reply with exactly one JSON object: {{"question": "...", "search_query": "...", "paraphrase": "..."}}"""
THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.S)


def question_word(stage, pid):
    words = stage.cfg.gen.question_words
    return words[int(pid.split(":")[1][:8], 16) % len(words)]


def build_messages(passage, qword):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(passage=passage, qword=qword)}]


class ServerBackend:
    def __init__(self, endpoint, model, api_key, g):
        self.url, self.model, self.g = endpoint.rstrip("/") + "/chat/completions", model, g
        self.headers = {"Content-Type": "application/json",
                        **({"Authorization": f"Bearer {api_key}"} if api_key else {})}
        self.local = threading.local()
        self.name = f"{model} at {endpoint}"

    def _session(self):
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
        return self.local.s

    def _one(self, messages):
        g = self.g
        body = {"model": self.model, "messages": messages, "temperature": g.temperature, "top_p": g.top_p,
                "max_tokens": g.max_tokens, **g.extra_body}
        if g.json_mode == "schema":
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "queries", "schema": JSON_SCHEMA, "strict": True}}
        elif g.json_mode == "object":
            body["response_format"] = {"type": "json_object"}
        error = ""
        for attempt in range(4):
            try:
                r = self._session().post(self.url, json=body, headers=self.headers, timeout=g.timeout_s)
                if r.status_code == 200:
                    j = r.json()
                    msg, usage = j["choices"][0]["message"], j.get("usage") or {}
                    return {"text": msg.get("content") or "",
                            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0), "error": ""}
                error = f"HTTP {r.status_code}: {r.text[:300]}"
                if r.status_code in (400, 401, 403, 404, 422):
                    break
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
        return {"text": "", "reasoning": "", "prompt_tokens": 0, "completion_tokens": 0, "error": error}

    def generate(self, batch):
        with ThreadPoolExecutor(self.g.concurrency) as pool:
            return list(pool.map(self._one, batch))

    def close(self):
        pass


class TransformersBackend:
    def __init__(self, model_name, g, device, batch_size=16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.g, self.batch_size, self.device = g, batch_size, device
        self.name = f"{model_name} (transformers, in this process)"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()

    @torch.no_grad()
    def generate(self, batch):
        out = []
        for i in range(0, len(batch), self.batch_size):
            prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False) for m in batch[i: i + self.batch_size]]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
            gen = self.model.generate(**enc, do_sample=True, temperature=self.g.temperature, top_p=self.g.top_p,
                                      max_new_tokens=self.g.max_tokens, pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            for j, ids in enumerate(new):
                n_new = int((ids != self.tok.pad_token_id).sum())
                out.append({"text": self.tok.decode(ids, skip_special_tokens=True), "reasoning": "",
                            "prompt_tokens": int(enc["attention_mask"][j].sum()), "completion_tokens": n_new,
                            "error": ""})
        return out

    def close(self):
        del self.model
        free_gpu()


def start_vllm_server(stage):
    g = stage.cfg.gen
    env = Path(g.vllm_env)
    vllm_bin = env / "bin" / "vllm"
    if not vllm_bin.exists():
        log(f"installing vLLM into {env} (a separate environment; a few minutes)")
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
        subprocess.run([str(env / "bin" / "pip"), "install", "-q", "--upgrade", "pip", "vllm"], check=True)
    version = subprocess.run([str(env / "bin" / "python"), "-c", "import vllm; print(vllm.__version__)"],
                             capture_output=True, text=True).stdout.strip()
    log_path = stage.work / "vllm_server.log"
    stage.work.mkdir(parents=True, exist_ok=True)
    cmd = [str(vllm_bin), "serve", g.model, "--port", str(g.vllm_port), *shlex.split(g.vllm_args)]
    log(f"starting vLLM {version}: {' '.join(cmd)}  (log: {log_path})")
    proc = subprocess.Popen(cmd, stdout=open(log_path, "a"), stderr=subprocess.STDOUT,
                            env={**os.environ, "HF_TOKEN": stage.token or ""})
    t0 = time.time()
    while time.time() - t0 < 3600:
        if proc.poll() is not None:
            tail = log_path.read_text(encoding="utf-8", errors="ignore")[-3000:]
            raise RuntimeError(f"the vLLM server stopped (exit {proc.returncode}). End of its log:\n{tail}")
        try:
            if requests.get(f"http://localhost:{g.vllm_port}/v1/models", timeout=5).status_code == 200:
                log(f"vLLM server ready after {time.time() - t0:.0f}s")
                return proc
        except requests.RequestException:
            pass
        time.sleep(10)
    raise TimeoutError("the vLLM server did not answer within an hour; see its log")


def make_backend(stage):
    g = stage.cfg.gen
    if g.backend == "transformers":
        return TransformersBackend(g.smoke_model if stage.smoke else g.model, g, stage.device)
    return ServerBackend(g.endpoint, g.api_model or g.model, os.environ.get(g.api_key_env), g)


def parse_output(text, reasoning=""):
    leak = "<think>" in text or bool(reasoning.strip())
    t = THINK_RE.sub("", text).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return None, "no JSON object", leak
    try:
        obj = json.loads(t[a: b + 1])
    except json.JSONDecodeError as e:
        return None, f"invalid JSON ({e.msg})", leak
    if not isinstance(obj, dict) or not all(isinstance(obj.get(k), str) and obj[k].strip() for k in QUERY_KEYS):
        return None, "a query is missing or empty", leak
    return {k: " ".join(obj[k].split()) for k in QUERY_KEYS}, "", leak


class Generator:
    def __init__(self, stage):
        self.stage = stage
        g = stage.cfg.gen
        self.vllm = None
        if g.backend == "server" and "localhost" in g.endpoint and not stage.smoke:
            self.vllm = start_vllm_server(stage)
        self.backend = make_backend(stage)
        log(f"generator: {self.backend.name}; thinking off; JSON mode: "
            f"{g.json_mode if g.backend == 'server' else 'none (plain generation, parsed afterwards)'}; "
            f"temperature {g.temperature}, top_p {g.top_p}, max {g.max_tokens} tokens")

    def generate_for(self, passages):
        t0 = time.time()
        results = self.backend.generate([build_messages(t, question_word(self.stage, p))
                                         for p, t in zip(passages["pid"], passages["text"])])
        seconds = time.time() - t0
        df = pd.DataFrame(results)
        df.insert(0, "pid", passages["pid"].to_numpy())
        return df, seconds

    def prompt_test(self, passages):
        stage, g = self.stage, self.stage.cfg.gen
        n_test = min(g.prompt_test_passages, len(passages))
        test, seconds = self.generate_for(passages.head(n_test))
        ok, lines = 0, []
        for i, (r, p) in enumerate(zip(test.itertuples(), passages.head(n_test).itertuples()), 1):
            queries, why, leak = parse_output(r.text, r.reasoning)
            ok += queries is not None
            lines.append(f"--- {i}. [{p.domain}] suggested word «{question_word(stage, p.pid)}»"
                         + ("  REASONING TEXT FOUND" if leak else ""))
            lines.append(f"  passage: {short(p.text, 420)}")
            if queries:
                lines += [f"  {k:<13} {queries[k]}" for k in QUERY_KEYS]
            else:
                lines.append(f"  PARSE FAILED ({why}{'; ' + r.error if r.error else ''}): {short(r.text, 300)!r}")
        ptok, ctok = test["prompt_tokens"].mean(), test["completion_tokens"].mean()
        stage.timings["gen_test"] = {"passages": n_test, "seconds": seconds, "prompt_tokens": float(ptok),
                                     "completion_tokens": float(ctok)}
        lines.append(f"{ok}/{n_test} parsed; {n_test / seconds:.2f} passages/s here with {self.backend.name}; "
                     f"{ptok:.0f} prompt + {ctok:.0f} output tokens per passage")
        log("\n".join(lines))
        if ok < g.min_parsed_share * n_test:
            stage.stop(f"prompt test: only {ok}/{n_test} outputs parsed (at least {g.min_parsed_share:.0%} needed). "
                       f"See the PARSE FAILED lines (and the vLLM log, {stage.work / 'vllm_server.log'}).")

    def run(self, passages):
        stage, g = self.stage, self.stage.cfg.gen
        gen_dir = stage.syn / "generation"
        n_shards = math.ceil(len(passages) / g.shard_passages)
        done = sum((gen_dir / f"shard-{k:05d}.parquet").exists() for k in range(n_shards))
        new_secs, new_n = 0.0, 0
        total = {"seconds": 0.0, "passages": 0, "prompt_tokens": 0, "completion_tokens": 0}
        stage.announce(f"generation: {len(passages):,} passages in {n_shards} shards, {done} done earlier")
        for k in range(n_shards):
            out = gen_dir / f"shard-{k:05d}.parquet"
            if out.exists():
                continue
            part = passages.iloc[k * g.shard_passages: (k + 1) * g.shard_passages]
            df, secs = self.generate_for(part)
            errors = int((df["error"] != "").sum())
            if errors > g.max_error_share * len(part):
                first = df.loc[df["error"] != "", "error"].iloc[0]
                stage.stop(f"generation shard {k + 1}/{n_shards}: {errors:,} of {len(part):,} requests failed, so it "
                           f"was not saved (finished shards are kept: start the server again and rerun). First error: "
                           f"{short(first, 300)}")
            write_parquet(df, out)
            done, new_secs, new_n = done + 1, new_secs + secs, new_n + len(part)
            total["seconds"] += secs
            total["passages"] += len(part)
            total["prompt_tokens"] += int(df["prompt_tokens"].sum())
            total["completion_tokens"] += int(df["completion_tokens"].sum())
            eta = (len(passages) - done * g.shard_passages) * new_secs / new_n
            stage.announce(f"generation shard {done}/{n_shards}: {len(part):,} passages in {secs:.0f}s "
                           f"({len(part) / secs:.1f}/s, {df['completion_tokens'].sum() / secs:,.0f} output tokens/s)"
                           + (f", {errors} request errors" if errors else "") + f"; ETA ~{fmt_hours(max(eta, 0))}")
        return total

    def close(self):
        self.backend.close()
        if self.vllm is not None:
            self.vllm.terminate()
            self.vllm.wait(timeout=120)
            log("vLLM server stopped")
        free_gpu()


def filter_step(name, df, reasons, details=None, n_examples=5, passage_texts=None):
    reasons = pd.Series(reasons, index=df.index, dtype=object)
    details = pd.Series(details if details is not None else [""] * len(df), index=df.index, dtype=object)
    hit = reasons.notna()
    lines = [f"{name}: rows in {len(df):,}, removed {int(hit.sum()):,}, rows out {int((~hit).sum()):,}"]
    lines += [f"  {r}: {n:,}" for r, n in reasons[hit].value_counts().items()]
    removed = df[hit].assign(reason=reasons[hit], detail=details[hit])
    if len(removed):
        lines.append(f"{min(n_examples, len(removed))} removed examples:")
        groups = [g for _, g in removed.groupby("reason")]
        ex = pd.concat([g.iloc[i: i + 1] for i in range(n_examples) for g in groups if i < len(g)]).head(n_examples)
        for r in ex.itertuples():
            text = r.query if "query" in removed.columns else short(r.text, 200)
            lines.append(f"  - {r.reason}" + (f" ({short(r.detail, 90)})" if r.detail else "") + f": {short(text, 160)}")
            if passage_texts is not None and "pid" in removed.columns and r.pid in passage_texts.index:
                lines.append(f"      passage: {short(passage_texts[r.pid], 170)}")
    log("\n".join(lines))
    return df[~hit]


def copied_run(stage, query, passage, n):
    q, p = WORD_RE.findall(stage.normalize(query)), WORD_RE.findall(stage.normalize(passage))
    grams = {tuple(p[i: i + n]) for i in range(len(p) - n + 1)}
    for i in range(len(q) - n + 1):
        if tuple(q[i: i + n]) in grams:
            return " ".join(q[i: i + n])
    return None


def banned_forms(stage, word):
    stem = stage.normalize(word)[2:]
    forms = {stem, stem[:-1] + "ه"} if stem.endswith("ة") else {stem, stem + "ة", stem + "ه"}
    return sorted(forms, key=len, reverse=True)


def banned_regex(stage):
    forms = "|".join(regex.escape(f) for w in stage.cfg.gen.banned_words for f in banned_forms(stage, w))
    return regex.compile(r"(?<!\p{L})[وف]?[بلك]?(?:ال|لل)(?:" + forms + r")(?!\p{L})")


def synthetic_queries(stage, generated, passages):
    g, f = stage.cfg.gen, stage.cfg.filters
    texts = passages.set_index("pid")["text"]
    parsed = [parse_output(t, rz) for t, rz in zip(generated["text"], generated["reasoning"])]
    leaks = sum(leak for _, _, leak in parsed)
    kept = filter_step("S5a JSON parse", generated,
                       [None if q else ("request failed" if e else f"parse failed: {why}")
                        for (q, why, _), e in zip(parsed, generated["error"])],
                       [short(t or e, 120) for t, e in zip(generated["text"], generated["error"])],
                       passage_texts=texts)
    log(f"reasoning text in the outputs: {leaks} of {len(generated):,}"
        + ("  <- thinking is NOT switched off: check extra_body / the chat template" if leaks else " (thinking is off)"))
    by_pid = {p: q for p, (q, _, _) in zip(generated["pid"], parsed) if q}
    syn_q = pd.DataFrame([(p, k, by_pid[p][k]) for p in kept["pid"] for k in QUERY_KEYS],
                         columns=["pid", "qtype", "query"])
    log(f"-> {len(syn_q):,} queries ({len(kept):,} passages x 3)")
    share = [arabic_share(stage.normalize(q)) for q in syn_q["query"]]
    syn_q = filter_step("S5b non-Arabic output", syn_q,
                        [f"< {f.min_arabic_letters:.0%} Arabic letters" if s < f.min_arabic_letters else None
                         for s in share], [f"{s:.0%} Arabic" for s in share], passage_texts=texts)
    limits = {"question": g.words_question, "paraphrase": g.words_question, "search_query": g.words_search}
    words = [len(q.split()) for q in syn_q["query"]]
    syn_q = filter_step("S5c lengths", syn_q,
                        [None if limits[t][0] <= n <= limits[t][1] else f"{t}: not {limits[t][0]}-{limits[t][1]} words"
                         for t, n in zip(syn_q["qtype"], words)], [f"{n} words" for n in words], passage_texts=texts)
    runs = [copied_run(stage, q, texts[p], g.copy_ngram) for q, p in zip(syn_q["query"], syn_q["pid"])]
    syn_q = filter_step("S5d copying", syn_q, [f"copies {g.copy_ngram}+ consecutive words" if r else None for r in runs],
                        [f"«{r}»" if r else "" for r in runs], passage_texts=texts)
    banned = banned_regex(stage)
    hits = [banned.search(stage.normalize(q)) for q in syn_q["query"]]
    syn_q = filter_step("S5e mentions the passage", syn_q,
                        ["mentions " + "/".join(g.banned_words) if h else None for h in hits],
                        [h.group(0) if h else "" for h in hits], passage_texts=texts)
    return syn_q


def build_synthetic_source(stage, syn_q, passages, generated, generator_name, gen_seconds):
    b = SourceBuilder(stage, "synthetic")
    b.t0 = time.time() - gen_seconds
    texts = passages.set_index("pid")["text"]
    doms = passages.set_index("pid")["domain"]
    for r in syn_q.itertuples():
        pid = b.passage(doms[r.pid], texts[r.pid], f"synthetic:{r.pid}")
        b.add(f"{r.pid}:{r.qtype}", r.query, [pid], group=pid, qtype=r.qtype, domain=doms[r.pid])
    b.stats.update({"passages": len(passages), "generated": len(generated), "queries kept": len(syn_q)})
    return b.save(f"generated by {generator_name} from {len(passages):,} passages (MIRACL-ar corpus + stage 3 news "
                  "leads)", stage.cfg.gen.n_passages)


def load_generated(stage):
    gen_dir = stage.syn / "generation"
    return pd.concat([pd.read_parquet(p) for p in sorted(gen_dir.glob("shard-*.parquet"))], ignore_index=True)
