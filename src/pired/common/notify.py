import contextlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .runlog import log


class TelegramNotifier:
    def __init__(self, token, chat_id, title):
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id, self.title = chat_id, title
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.last_error = 0.0

    def send(self, text):
        self.pool.submit(self._post, f"{self.title}\n{text}"[:4000])

    def flush(self, timeout=20.0):
        with contextlib.suppress(Exception):
            self.pool.submit(lambda: None).result(timeout=timeout)

    def _post(self, text):
        body = json.dumps({"chat_id": self.chat_id, "text": text}).encode("utf-8")
        request = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        error = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=15):
                    return
            except Exception as e:
                error = e
                time.sleep(5 * (attempt + 1))
        if time.time() - self.last_error > 600:
            self.last_error = time.time()
            print(f"[telegram] message failed ({type(error).__name__}: {error}); the run continues", flush=True)


def make_notifier(title, enabled=True):
    token, chat_id = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    return TelegramNotifier(token, chat_id, title) if enabled and token and chat_id else None


class Announcer:
    def __init__(self, title, enabled=True):
        self.notifier = make_notifier(title, enabled)

    def __call__(self, text):
        log(text)
        if self.notifier is not None:
            self.notifier.send(text)

    def stop(self, message):
        self(f"STOPPED: {message}")
        if self.notifier is not None:
            self.notifier.flush()
        raise RuntimeError(message)
