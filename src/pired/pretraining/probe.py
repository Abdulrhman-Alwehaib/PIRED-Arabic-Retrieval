import math
import re
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from ..tokenizer.normalization import METASPACE
from .collator import MLMCollator
from .config import MLMSettings

PLACEHOLDER = "كلمة"
MASK_TYPOS = re.compile(re.escape("[") + " *mask *[" + re.escape("]") + "}]", re.IGNORECASE)

EXAMPLES = [
    "المملكة العربية السعودية عاصمتها [MASK].",
    "بسم الله الرحمن [MASK].",
    "ذهبت إلى [MASK] لأشتري الخبز والحليب.",
    "المملكة [MASK] [MASK] عاصمتها الرياض.",
]

QUIZ = [
    ("المملكة العربية السعودية عاصمتها [MASK].", ["الرياض"]),
    ("عاصمة مصر هي [MASK].", ["القاهرة"]),
    ("عاصمة فرنسا هي [MASK].", ["باريس"]),
    ("عاصمة اليابان هي [MASK].", ["طوكيو"]),
    ("عاصمة العراق هي [MASK].", ["بغداد"]),
    ("عاصمة سوريا هي [MASK].", ["دمشق"]),
    ("عاصمة قطر هي [MASK].", ["الدوحة"]),
    ("عاصمة الإمارات العربية المتحدة هي [MASK].", ["أبوظبي"]),
    ("يقع المسجد النبوي في المدينة [MASK].", ["المنورة"]),
    ("تقع الكعبة المشرفة في [MASK] المكرمة.", ["مكة"]),
    ("يصوم المسلمون في شهر [MASK].", ["رمضان"]),
    ("نزل القرآن الكريم باللغة [MASK].", ["العربية"]),
    ("بسم الله الرحمن [MASK].", ["الرحيم"]),
    ("السلام عليكم ورحمة الله [MASK].", ["وبركاته"]),
    ("صلاة [MASK] هي أول صلاة في اليوم.", ["الفجر"]),
    ("يوم [MASK] هو يوم العطلة الأسبوعية عند المسلمين.", ["الجمعة"]),
    ("أطول نهر في العالم هو نهر [MASK].", ["النيل"]),
    ("تشرق الشمس من [MASK] وتغرب من الغرب.", ["الشرق"]),
    ("عدد أيام الأسبوع [MASK] أيام.", ["سبعة"]),
    ("فصل [MASK] هو أحر فصول السنة.", ["الصيف"]),
    ("يدور القمر حول [MASK].", ["الأرض"]),
    ("تدور الأرض حول [MASK].", ["الشمس"]),
    ("يشرب الإنسان [MASK] عندما يشعر بالعطش.", ["الماء"]),
    ("ذهب الطالب إلى [MASK] ليتعلم القراءة والكتابة.", ["المدرسة"]),
    ("نقل المريض إلى [MASK] لتلقي العلاج.", ["المستشفى"]),
    ("كرة [MASK] هي الرياضة الأكثر شعبية في العالم.", ["القدم"]),
    ("برج [MASK] في دبي هو أطول مبنى في العالم.", ["خليفة"]),
    ("العملة الرسمية في المملكة العربية السعودية هي [MASK] السعودي.", ["الريال"]),
    ("يقع الخليج [MASK] شرق المملكة العربية السعودية.", ["العربي"]),
    ("يسكن رئيس الولايات المتحدة في البيت [MASK].", ["الأبيض"]),
    ("ذهبت [MASK] السوق لشراء الخضار.", ["إلى"]),
    ("الأسد ملك [MASK].", ["الغابة"]),
]


class MaskFiller:
    def __init__(self, model, tokenizer, device):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.mask = tokenizer.mask_token
        self.special_ids = set(tokenizer.all_special_ids)

    def encode_with_masks(self, text):
        tok = self.tokenizer
        text = MASK_TYPOS.sub(self.mask, text)
        pieces = text.split(self.mask)
        if len(pieces) < 2:
            raise ValueError(f"put {self.mask} where the missing word goes")
        joined, spans = "", []
        for i, piece in enumerate(pieces):
            joined += piece
            if i < len(pieces) - 1:
                spans.append((len(joined), len(joined) + len(PLACEHOLDER)))
                joined += PLACEHOLDER
        enc = tok(joined, return_offsets_mapping=True)
        ids, offsets = enc["input_ids"], enc["offset_mapping"]
        groups = [[i for i, (s, e) in enumerate(offsets) if ids[i] not in self.special_ids and s < end and e > start]
                  for start, end in spans]
        if any(not g for g in groups) or len({i for g in groups for i in g}) < sum(map(len, groups)):
            raise ValueError(f"separate the {self.mask} tokens from each other with spaces")
        first, rest = {g[0] for g in groups}, {i for g in groups for i in g[1:]}
        out, positions = [], []
        for i, token_id in enumerate(ids):
            if i in rest:
                continue
            if i in first:
                positions.append(len(out))
                token_id = tok.mask_token_id
            out.append(token_id)
        return out, positions

    def token_word(self, token_id):
        token = self.tokenizer.convert_ids_to_tokens(token_id)
        return token[1:] if token.startswith(METASPACE) else "…" + token

    @torch.no_grad()
    def predict(self, text, top_k=5, model=None):
        model = self.model if model is None else model
        ids, positions = self.encode_with_masks(text)
        logits = model(torch.tensor([ids], device=self.device)).logits[0, positions].float()
        top = logits.softmax(-1).topk(top_k, dim=-1)
        guesses = [[(self.token_word(t), p) for t, p in zip(ti.tolist(), pi.tolist())]
                   for ti, pi in zip(top.indices, top.values)]
        filled = list(ids)
        for pos, best in zip(positions, top.indices[:, 0].tolist()):
            filled[pos] = best
        return self.tokenizer.decode(filled, skip_special_tokens=True), guesses

    def describe(self, text, top_k=5, model=None):
        sentence, guesses = self.predict(text, top_k, model)
        lines = [text, f"-> {sentence}"]
        for k, options in enumerate(guesses, 1):
            lines.append(f"   [MASK] {k}: " + ", ".join(f"{w} ({p:.0%})" for w, p in options))
        return "\n".join(lines)

    def normalize(self, word):
        return self.tokenizer.backend_tokenizer.normalizer.normalize_str(word).strip()

    def quiz(self, quiz=QUIZ, model=None):
        top1 = top5 = 0
        rows = []
        for sentence, answers in quiz:
            _, guesses = self.predict(sentence, 5, model)
            words = [self.normalize(w) for w, _ in guesses[0]]
            accepted = {self.normalize(a) for a in answers}
            hit1, hit5 = words[0] in accepted, bool(accepted & set(words))
            top1, top5 = top1 + hit1, top5 + hit5
            rows.append({"sentence": sentence, "answer": answers[0], "guesses": words,
                         "result": "top1" if hit1 else "top5" if hit5 else "miss"})
        return {"top1": top1 / len(quiz), "top5": top5 / len(quiz), "rows": rows}


def load_validation_rows(local_file, repo_id=None, token=None, n_rows=None):
    local_file = Path(local_file)
    if local_file.exists():
        return np.load(local_file)[:n_rows]
    if repo_id is None:
        raise FileNotFoundError(f"{local_file} is missing and no Hub dataset repo is given")
    return np.load(hf_hub_download(repo_id, "validation.npy", repo_type="dataset", token=token))[:n_rows]


@torch.no_grad()
def heldout_accuracy(model, tokenizer, rows, device, batch_size=32, forward_rows=8):
    collator = MLMCollator.from_tokenizer(tokenizer, MLMSettings(), seed=0)
    loss_sum = top1 = top5 = n = 0
    for i in range(0, len(rows), batch_size):
        batch = collator(rows[i: i + batch_size])
        for j in range(0, len(batch["input_ids"]), forward_rows):
            part = {k: v[j: j + forward_rows].to(device) for k, v in batch.items()}
            out = model(**part)
            targets = part["labels"][part["labels"] != -100]
            best = out.logits.float().topk(5, dim=-1).indices
            loss_sum += out.loss.item() * len(targets)
            top1 += (best[:, 0] == targets).sum().item()
            top5 += (best == targets[:, None]).any(dim=-1).sum().item()
            n += len(targets)
    return {"top1": top1 / n, "top5": top5 / n, "loss": loss_sum / n, "perplexity": math.exp(loss_sum / n),
            "masked_tokens": n}
