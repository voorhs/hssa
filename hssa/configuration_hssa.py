from transformers.models.mpnet.configuration_mpnet import MPNetConfig


class HSSAConfig(MPNetConfig):
    def __init__(
        self,
        vocab_size = 30529, # additional_special_tokens, see tokenization_hssa.py
        max_turn_embeddings=36,
        **kwargs,
    ):
        self.max_turn_embeddings = max_turn_embeddings
        super().__init__(**kwargs)
