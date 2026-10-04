import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput

from .configuration import ArabicEncoderConfig

try:
    from transformers import initialization as hf_init
except ImportError:
    from torch.nn import init as hf_init


def rotary_cos_sin(seq_len: int, head_dim: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = torch.arange(seq_len, device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_f = x.float()
    x1, x2 = x_f.chunk(2, dim=-1)
    return (x_f * cos + torch.cat((-x2, x1), dim=-1) * sin).to(x.dtype)


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    hidden = hidden.float()
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class ArabicEncoderEmbeddings(nn.Module):
    def __init__(self, config: ArabicEncoderConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.tok_embeddings(input_ids)))


class ArabicEncoderAttention(nn.Module):
    def __init__(self, config: ArabicEncoderConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout_prob
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj.is_residual_projection = True

    def forward(self, x, cos, sin, attn_mask):
        batch, seq, _ = x.shape
        q, k, v = self.qkv(x).view(batch, seq, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.dropout if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).reshape(batch, seq, -1))


class ArabicEncoderMLP(nn.Module):
    def __init__(self, config: ArabicEncoderConfig):
        super().__init__()
        self.wi = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.wo.is_residual_projection = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        a, b = self.wi(x).chunk(2, dim=-1)
        return self.wo(self.dropout(F.gelu(a) * b))


class ArabicEncoderLayer(nn.Module):
    def __init__(self, config: ArabicEncoderConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = ArabicEncoderAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = ArabicEncoderMLP(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.dropout(self.attn(self.attn_norm(x), cos, sin, attn_mask))
        return x + self.dropout(self.mlp(self.mlp_norm(x)))


class ArabicEncoderPreTrainedModel(PreTrainedModel):
    config_class = ArabicEncoderConfig
    base_model_prefix = "model"
    _no_split_modules = ["ArabicEncoderLayer"]
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            if getattr(module, "is_residual_projection", False):
                std = std / math.sqrt(2 * self.config.num_hidden_layers)
            hf_init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                hf_init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            hf_init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            hf_init.ones_(module.weight)
            if module.bias is not None:
                hf_init.zeros_(module.bias)


class ArabicEncoderModel(ArabicEncoderPreTrainedModel):
    def __init__(self, config: ArabicEncoderConfig):
        super().__init__(config)
        self.embeddings = ArabicEncoderEmbeddings(config)
        self.layers = nn.ModuleList(ArabicEncoderLayer(config) for _ in range(config.num_hidden_layers))
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value

    def forward(self, input_ids, attention_mask=None, output_hidden_states: bool = False, return_dict=None):
        seq_len = input_ids.shape[1]
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(f"sequence length {seq_len} > max_position_embeddings {self.config.max_position_embeddings}")
        cos, sin = rotary_cos_sin(seq_len, self.config.head_dim, self.config.rope_theta, input_ids.device)
        attn_mask = None if attention_mask is None else attention_mask[:, None, None, :].bool()
        x = self.embeddings(input_ids)
        hidden_states = [x] if output_hidden_states else None
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, attn_mask)
            if output_hidden_states and i < len(self.layers) - 1:
                hidden_states.append(x)
        x = self.final_norm(x)
        if output_hidden_states:
            hidden_states.append(x)
        out = BaseModelOutput(last_hidden_state=x, hidden_states=tuple(hidden_states) if hidden_states else None)
        return out.to_tuple() if return_dict is False else out

    def embed(self, input_ids, attention_mask=None) -> torch.Tensor:
        hidden = self(input_ids, attention_mask).last_hidden_state
        return F.normalize(mean_pool(hidden, attention_mask), p=2, dim=-1)


class ArabicEncoderMLMHead(nn.Module):
    def __init__(self, config: ArabicEncoderConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.norm(F.gelu(self.dense(hidden))))


class ArabicEncoderForMaskedLM(ArabicEncoderPreTrainedModel):
    _tied_weights_keys = {"head.decoder.weight": "model.embeddings.tok_embeddings.weight"}

    def __init__(self, config: ArabicEncoderConfig):
        super().__init__(config)
        self.model = ArabicEncoderModel(config)
        self.head = ArabicEncoderMLMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.head.decoder = new_embeddings

    def forward(self, input_ids, attention_mask=None, labels=None, num_items_in_batch=None, return_dict=None):
        hidden = self.model(input_ids, attention_mask).last_hidden_state
        if labels is None:
            out = MaskedLMOutput(logits=self.head(hidden))
        else:
            masked = labels != -100
            logits = self.head(hidden[masked])
            targets = labels[masked]
            loss = F.cross_entropy(logits, targets, reduction="sum")
            denominator = num_items_in_batch if num_items_in_batch is not None else targets.numel()
            out = MaskedLMOutput(loss=loss / max(denominator, 1), logits=logits)
        return out.to_tuple() if return_dict is False else out

    def save_encoder(self, save_directory, **kwargs):
        self.model.save_pretrained(save_directory, **kwargs)
