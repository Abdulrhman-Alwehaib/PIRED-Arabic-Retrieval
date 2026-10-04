import math

import numpy as np
import torch


class MLMCollator:
    def __init__(self, vocab_size, pad_token_id, mask_token_id, special_token_ids, settings, seed=0):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.settings = settings
        self.special_ids = torch.tensor(sorted(set(special_token_ids) | {pad_token_id}), dtype=torch.long)
        regular = torch.ones(vocab_size, dtype=torch.bool)
        regular[self.special_ids] = False
        self.regular_ids = regular.nonzero().squeeze(1)
        self.generator = torch.Generator().manual_seed(seed)

    @classmethod
    def from_tokenizer(cls, tokenizer, settings, seed=0):
        return cls(len(tokenizer), tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.all_special_ids,
                   settings, seed)

    def __call__(self, sequences):
        if isinstance(sequences, np.ndarray):
            return self.mask_tokens(torch.from_numpy(sequences.astype(np.int64)))
        longest = max(len(s) for s in sequences)
        multiple = self.settings.pad_to_multiple_of
        width = math.ceil(longest / multiple) * multiple if multiple else longest
        input_ids = torch.full((len(sequences), width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, seq in enumerate(sequences):
            input_ids[i, : len(seq)] = torch.as_tensor(seq, dtype=torch.long)
            attention_mask[i, : len(seq)] = 1
        return self.mask_tokens(input_ids, attention_mask)

    def mask_tokens(self, input_ids, attention_mask=None):
        s, g = self.settings, self.generator
        input_ids = input_ids.clone()
        maskable = ~torch.isin(input_ids, self.special_ids)
        if attention_mask is not None:
            maskable &= attention_mask.bool()
        selected = (torch.rand(input_ids.shape, generator=g) < s.mlm_probability) & maskable
        labels = torch.where(selected, input_ids, torch.full_like(input_ids, -100))
        action = torch.rand(input_ids.shape, generator=g)
        to_mask = selected & (action < s.mask_replace_prob)
        to_random = selected & (action >= s.mask_replace_prob) & (action < s.mask_replace_prob + s.random_replace_prob)
        random_ids = self.regular_ids[torch.randint(len(self.regular_ids), input_ids.shape, generator=g)]
        input_ids[to_mask] = self.mask_token_id
        input_ids[to_random] = random_ids[to_random]
        batch = {"input_ids": input_ids, "labels": labels}
        if attention_mask is not None and not bool(attention_mask.all()):
            batch["attention_mask"] = attention_mask
        return batch

    def state_dict(self):
        return {"generator": self.generator.get_state()}

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"])
