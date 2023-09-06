from transformers.models.mpnet.configuration_mpnet import MPNetConfig


class HSSAConfig(MPNetConfig):
    def __init__(
        self,
        max_turn_embeddings=36,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_turn_embeddings = max_turn_embeddings
