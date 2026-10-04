import hashlib

import numpy as np


def short(text, n=160):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def md_cell(value):
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value).replace("|", "\\|")


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
                      *["| " + " | ".join(md_cell(v) for v in row) + " |" for row in rows]])


def fmt_hours(seconds):
    return f"{seconds / 60:.1f} min" if seconds < 3600 else f"{seconds / 3600:.1f} h"


def text_key(normalized_text):
    return hashlib.blake2b(" ".join(normalized_text.split()).encode("utf-8"), digest_size=8).hexdigest()
