import json
import time
from pathlib import Path


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


class JsonlLogger:
    def __init__(self, path, echo=True, notifier=None):
        self.path, self.echo, self.notifier = Path(path), echo, notifier
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record):
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
        payload = json.dumps(record, ensure_ascii=False, default=str)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(payload + "\n")
        if self.echo:
            print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()
                             if not isinstance(v, dict)),
                  flush=True)
        if self.notifier is not None:
            self.notifier.send(("ERROR " if record.get("event") == "error" else "") + payload)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
