from typing import Optional, Tuple
from transformers.models.roberta_prelayernorm.modeling_roberta_prelayernorm import (
    RobertaPreLayerNormSelfAttention as SelfAttention,
    RobertaPreLayerNormSelfOutput as SelfOutput,
    RobertaPreLayerNormIntermediate as Intermediate,
    RobertaPreLayerNormOutput as Output,
    RobertaPreLayerNormLayer as Layer,
    RobertaPreLayerNormEncoder as Encoder,
    RobertaPreLayerNormModel as Model
)

from configuration_hssa import HSSAConfig
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


class HSSAAttention(nn.Module):
    def __init__(self, config: HSSAConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self = SelfAttention(config)
        self.bpooler = SegmentPooler(config)
        self.updater = SegmentUpdater(config)
        self.output = SelfOutput(config)
        
    def forward(
        self,
        hidden_states,
        attention_mask,
        segment_size,
        utterance_mask,
        **kwargs
    ):
        """
        hidden_states: (B*S, T, d), states for each token
        attention_mask: (B*S, T)
        """
         
        # (B*S, T, d), attention within utterance
        token_states = self.self(hidden_states, attention_mask=attention_mask)
        _, T, d = token_states.shape

        # (B*S, d), using pooling method to get segment repr
        utterance_states = self.bpooler(token_states, attention_mask)

        # (B, S, d)
        utterance_states = utterance_states.view(-1, segment_size, d)
        B, S, _ = utterance_states.shape

        # utterances iteraction 
        utterance_states = self.self(
            utterance_states,
            attention_mask=utterance_mask
        )

        # (B*S, T, d), update the token hidden states with corresponding utterance states
        token_states = self.updater(
            utterance_states,
            token_states,
            attention_mask
        )

        # (B, S*T, d)
        token_states = token_states.view(B, S * T, d)
        flowed_states = self.output(token_states, hidden_states)

        return flowed_states


# instead of RobertaLayer
class HSSALayer(Layer):
    def __init__(self, config: HSSAConfig):
        super().__init__(config)

        # overrides RobertaLayer's attributes
        self.attention = HSSAAttention(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)

    def forward(
            self,
            hidden_states,
            attention_mask,
            segment_size,
            utterance_mask,
            **kwargs
        ):
        return super().forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)

class HSSAEncoder(Encoder):
    def __init__(self, config):
        super().__init__(config)

        self.layer = nn.ModuleList([HSSALayer(config) for _ in range(config.num_hidden_layers)])


class HSSAModel(Model):
    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)

        self.encoder = HSSAEncoder(config)

