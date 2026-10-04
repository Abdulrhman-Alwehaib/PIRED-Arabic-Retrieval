from transformers import PretrainedConfig

MODEL_TYPE = "arabic_encoder"


class ArabicEncoderConfig(PretrainedConfig):
    model_type = MODEL_TYPE

    def __init__(
        self,
        vocab_size: int = 64_000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 1024,
        rope_theta: float = 10_000.0,
        layer_norm_eps: float = 1e-5,
        hidden_dropout_prob: float = 0.0,
        attention_dropout_prob: float = 0.0,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        cls_token_id: int = 2,
        sep_token_id: int = 3,
        mask_token_id: int = 4,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        if hidden_size % num_attention_heads:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by num_attention_heads {num_attention_heads}")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.initializer_range = initializer_range
        kwargs.pop("head_dim", None)
        super().__init__(
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            mask_token_id=mask_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
