from transformers.models.mpnet.configuration_mpnet import MPNetConfig


class HSSAConfig(MPNetConfig):
    def __init__(
        self,
        max_turn_embeddings=None,
        max_position_embeddings=514,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_turn_embeddings = max_turn_embeddings
        self.max_position_embeddings = max_position_embeddings
