from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

from .configuration import MODEL_TYPE, ArabicEncoderConfig
from .embedding import PASSAGE_PREFIX, QUERY_PREFIX, Embedder, load_tokenizer
from .modeling import ArabicEncoderForMaskedLM, ArabicEncoderModel, mean_pool
from .weights import (check_reference_outputs, load_encoder, load_masked_lm, load_weights_strict,
                      save_reference_outputs)

AutoConfig.register(MODEL_TYPE, ArabicEncoderConfig, exist_ok=True)
AutoModel.register(ArabicEncoderConfig, ArabicEncoderModel, exist_ok=True)
AutoModelForMaskedLM.register(ArabicEncoderConfig, ArabicEncoderForMaskedLM, exist_ok=True)

__all__ = [
    "MODEL_TYPE",
    "PASSAGE_PREFIX",
    "QUERY_PREFIX",
    "ArabicEncoderConfig",
    "ArabicEncoderForMaskedLM",
    "ArabicEncoderModel",
    "Embedder",
    "check_reference_outputs",
    "load_encoder",
    "load_masked_lm",
    "load_tokenizer",
    "load_weights_strict",
    "mean_pool",
    "save_reference_outputs",
]
