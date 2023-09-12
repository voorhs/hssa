from transformers.models.mpnet.modeling_mpnet import (
    MPNetSelfAttention as SelfAttention,
    MPNetIntermediate as Intermediate,
    MPNetOutput as Output,
    MPNetPreTrainedModel as PreTrainedModel,
)
from typing import Optional
from .configuration_hssa import HSSAConfig
import torch.nn as nn
import torch
import math


class SegmentPooler(nn.Module):
    def __init__(self, config: HSSAConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        
    def forward(self, token_states: torch.Tensor, attention_mask: torch.Tensor):
        """
        Pool segments

        Params
        ------
            - `token_states`: (B*S, T, d)
            - `attention_mask`: (B*S, T), attention mask for each utterance

        Return
        ------
            - `utterance_states`: (B*S, d)
            - `utterance_mask`: (B*S, 1)
        
        Notation
        --------
        B: batch size
        S: segments number
        d: hidden size
        T: size of segment
        """

        _, T, hidden_size = token_states.shape

        # (B*S, T, 1)
        attention_mask = attention_mask.view(-1, T, 1)

        # (B*S, d)
        avg_tok_states = torch.sum(attention_mask * token_states, dim=1) / (1e-6 + attention_mask.sum(dim=1))

        # (B*S, d, 1)
        avg_tok_states = avg_tok_states.unsqueeze(2) 

        # (B*S, T, d) x (B*S, d, 1) -> (B*S, T, 1)
        scores = torch.bmm(token_states, avg_tok_states) / math.sqrt(hidden_size)
        scores += attention_mask
        scores = torch.softmax(scores, dim=1)

        # (B*S, T, d) * (B*S, T, 1) -> (B*S, d)
        utterance_states = torch.sum(token_states * scores, dim=1)
        utterance_states = self.dense(utterance_states)

        return utterance_states


class SegmentUpdater(nn.Module):
    def __init__(self, config: HSSAConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

    def forward(
        self,
        utterance_states: torch.Tensor,
        token_states: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        """
        Update hidden states of each token in dialogue.

        utterance_states: (B*S, d)
        token_states: (B*S, T, d)
        attention_mask: (B*S, T)

        where
        B: batch size
        S: segments number
        d: hidden size
        T: size of segment
        """

        # (B*S, d, 1)
        utterance_states = utterance_states.view(-1, self.hidden_size, 1)

        # (B*S, T, d) x (B*S, d, 1) -> (B*S, T, 1)
        scores = torch.bmm(token_states, utterance_states) / math.sqrt(self.hidden_size)
        scores += (1 - attention_mask).mul(-1e5).exp().unsqueeze(2)
        scores = torch.softmax(scores, dim=1)

        # (B*S, T, d) + (B*S, T, 1) x (B*S, 1, d)
        hidden_states = token_states +  scores * utterance_states.view(-1, 1, self.hidden_size)

        # (B*S, T, d)
        return hidden_states


def get_extended_attention_mask(attention_mask):
    if len(attention_mask.shape) == 2:
        return attention_mask[:, None, None, :]
    elif len(attention_mask.shape) == 3:
        return attention_mask[:, None, :, :]


# instead of MPNetAttention
class HSSAAttention(nn.Module):
    def __init__(self, config: HSSAConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn = SelfAttention(config)
        self.bpooler = SegmentPooler(config)
        self.updater = SegmentUpdater(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states,
        attention_mask,
        max_dia_len,
        utterance_mask,
        position_bias
    ):
        """
        hidden_states: (B*S, T, d), states for each token
        attention_mask: (B*S, T)
        max_dia_len: S
        utterance_mask: (B, S)
        """
         
        # (B*S, T, d), attention within utterance
        token_states = self.attn(
            hidden_states,
            attention_mask=get_extended_attention_mask(attention_mask),
            position_bias=position_bias
        )[0]
        _, T, d = token_states.shape

        # (B*S, d), using pooling method to get segment repr
        utterance_states = self.bpooler(token_states, attention_mask)

        # (B, S, d)
        utterance_states = utterance_states.view(-1, max_dia_len, d)
        B, S, _ = utterance_states.shape

        # utterances iteraction 
        utterance_states = self.attn(
            utterance_states,
            attention_mask=get_extended_attention_mask(utterance_mask)
        )[0]

        # (B*S, T, d), update the token hidden states with corresponding utterance states
        token_states = self.updater(
            utterance_states,
            token_states,
            attention_mask
        )

        token_states = self.LayerNorm(token_states)

        return token_states


# a little modified MPNetLayer
class HSSALayer(nn.Module):
    def __init__(self, config: HSSAConfig):
        super().__init__()
        self.attention = HSSAAttention(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)

    def forward(
            self,
            hidden_states,
            attention_mask,
            max_dia_len,
            utterance_mask,
            position_bias
        ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            max_dia_len,
            utterance_mask,
            position_bias
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# much reduced MPNetEncoder
class HSSAEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_heads = config.num_attention_heads
        self.layer = nn.ModuleList([HSSALayer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets, self.n_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        max_dia_len,
        utterance_mask,
    ):
        position_bias = self.compute_position_bias(hidden_states)
        for layer_module in self.layer:
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                max_dia_len,
                utterance_mask,
                position_bias=position_bias,
            )

        return hidden_states

    def compute_position_bias(self, x, position_ids=None, num_buckets=32):
        bsz, qlen, klen = x.size(0), x.size(1), x.size(1)
        if position_ids is not None:
            context_position = position_ids[:, :, None]
            memory_position = position_ids[:, None, :]
        else:
            context_position = torch.arange(qlen, dtype=torch.long)[:, None]
            memory_position = torch.arange(klen, dtype=torch.long)[None, :]

        relative_position = memory_position - context_position

        rp_bucket = self.relative_position_bucket(relative_position, num_buckets=num_buckets)
        rp_bucket = rp_bucket.to(x.device)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        values = values.expand((bsz, -1, qlen, klen)).contiguous()
        return values

    @staticmethod
    def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)

        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret


# a little modified MPNetEmbeddings
class HSSAEmbeddings(nn.Module):
    def __init__(self, config: HSSAConfig):
        super().__init__()
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        if config.max_turn_embeddings is not None:
            self.turn_embeddings = nn.Embedding(
                config.max_turn_embeddings, config.hidden_size, padding_idx=self.padding_idx
            )
            self.register_buffer(
                "turn_ids", torch.arange(config.max_turn_embeddings), persistent=False
            )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.config = config

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, max_dia_len=None, **kwargs):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        
        if self.config.max_turn_embeddings is not None:
            B = input_shape[0] // max_dia_len
            turn_embeddings = self.turn_embeddings(self.turn_ids[:max_dia_len])[None, :, None, :]
            embeddings = embeddings.view(B, max_dia_len, seq_length, -1) + turn_embeddings
            embeddings = embeddings.view(-1, seq_length, self.config.hidden_size)
            
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class HSSAPreTrainedModel(PreTrainedModel):
    config_class = HSSAConfig
    base_model_prefix = "mpnet"


# much reduced MPNetModel
class HSSAModel(HSSAPreTrainedModel):
    def __init__(self, config: HSSAConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = HSSAEmbeddings(config)
        self.encoder = HSSAEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        attention_mask: Optional[torch.FloatTensor],
        max_dia_len,
        utterance_mask,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            max_dia_len=max_dia_len
        )
        hidden_states = self.encoder(
            embedding_output,
            attention_mask,
            max_dia_len,
            utterance_mask
        )

        return hidden_states

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


# helper for Embedding
def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`. :param torch.Tensor x: :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx