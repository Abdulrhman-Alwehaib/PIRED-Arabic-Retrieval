import json
import os
from functools import cached_property
from pathlib import Path

from huggingface_hub import HfApi, get_token

SMOKE_FOLDERS = {
    "tokenizer_dir": "1_tokenizer",
    "tokenizer_corpus_dir": "1_tokenizer/data",
    "pretraining_data_dir": "2_pretraining_data",
    "mlm_stage_dir": "2_mlm_pretraining",
    "contrastive_weak_stage_dir": "3_contrastive_weak",
    "supervised_stage_dir": "4_supervised",
    "evaluation_stage_dir": "5_evaluation",
}


def find_project_root(start=None):
    start = Path(start or os.environ.get("PIRED_ROOT") or Path.cwd()).resolve()
    for path in (start, *start.parents):
        if (path / "project.json").exists():
            return path
    raise FileNotFoundError("project.json not found here or in any parent directory")


def load_dotenv_file(path):
    path = Path(path)
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip().strip("'\"")
        if value:
            os.environ.setdefault(key.strip(), value)


class Project:
    def __init__(self, root=None, smoke=False):
        self.root = find_project_root(root)
        load_dotenv_file(self.root / ".env")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        self.settings = json.loads((self.root / "project.json").read_text(encoding="utf-8"))
        self.name = self.settings["project_name"]
        self.model_type = self.settings["model_type"]
        self.smoke = smoke

    def path(self, key):
        return self.root / self.settings["paths"][key]

    def stage_dir(self, key):
        return self.path("smoke_tests_dir") / SMOKE_FOLDERS[key] if self.smoke else self.path(key)

    @property
    def hf_token(self):
        return os.environ.get("HF_TOKEN") or get_token()

    @cached_property
    def api(self):
        return HfApi(token=self.hf_token)

    @cached_property
    def namespace(self):
        return self.settings.get("hf_namespace") or self.api.whoami()["name"]

    def repo_id(self, suffix, explicit=None, smoke_suffix=False):
        if explicit:
            return explicit
        return f"{self.namespace}/{self.name}-{suffix}{'-smoke' if smoke_suffix and self.smoke else ''}"
