import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import torch

from .runlog import log


def fork_available():
    return "fork" in mp.get_all_start_methods()


def _worker_init():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)


def parallel_map(fn, items, workers, desc="", every=120.0):
    items = list(items)
    workers = min(workers, len(items))
    results, start, last = [None] * len(items), time.time(), time.time()
    if workers > 1 and fork_available():
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"), initializer=_worker_init) as pool:
            futures = {pool.submit(fn, x): i for i, x in enumerate(items)}
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=every, return_when=FIRST_COMPLETED)
                for future in done:
                    results[futures[future]] = future.result()
                if desc and time.time() - last >= every:
                    last = time.time()
                    log(f"{desc}: {len(items) - len(pending)}/{len(items)} ({(last - start) / 60:.1f} min)")
    else:
        for i, x in enumerate(items):
            results[i] = fn(x)
            if desc and time.time() - last >= every:
                last = time.time()
                log(f"{desc}: {i + 1}/{len(items)} ({(last - start) / 60:.1f} min)")
    return results
