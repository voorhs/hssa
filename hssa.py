from dataclasses import dataclass
import copy
import torch
import torch.nn as nn
import math
from typing import Tuple
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import (
    PreTrainedModel,
    PretrainedConfig,
    apply_chunking_to_forward,
    prune_linear_layer,
    find_pruneable_heads_and_indices,
    SPIECE_UNDERLINE
)
from transformers.activations import ACT2FN
from transformers import BertTokenizerFast, PreTrainedTokenizerFast


class FlowBertConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,

        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        position_embedding_type="absolute",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        classifier_dropout=0.2,

        vocab_size=21128,
        max_position_embeddings=512,
        type_vocab_size=2,
        pad_token_id=0,

        segment_size=[8, 16, 32, 32, 64, 64, 64, 128, 128, 128],#
        jdembedding=False,#
        max_turn_embeddings=36,#
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            **kwargs,
        )
        
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.max_turn_embeddings = max_turn_embeddings
        self.type_vocab_size = type_vocab_size
        self.position_embedding_type = position_embedding_type

        self.adapter_size = adapter_size
        self.lst_ratio = lst_ratio
        self.gate_temp  = gate_temp
        self.gate_alpha = gate_alpha
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size if embedding_size is not None else hidden_size
        self.intermediate_size = intermediate_size
        self.frozen_layers = frozen_layers
        self.frozen_embeddings =frozen_embeddings
        self.tasks=tasks
        
        self.seq_length =seq_length

        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout = classifier_dropout

        self.jdembedding = jdembedding
        self.segment_size =segment_size

        self.num_conv_layers = num_conv_layers 
        self.conv_kernel_size=conv_kernel_size
        self.kernel_num = self.num_attention_heads

        self.temperature = temperature
        self.membank= membank
    
    def check_config(self):
         for i, blsz in enumerate(self.segment_size):
            assert self.seq_length % blsz ==0, \
                "seq_length must be multiplication of segment_size"+ f"{self.seq_length}/{blsz}({i})"


class DialEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config: FlowBertConfig):
        super().__init__()
        embedding_size = getattr(config, 'embedding_size', config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, embedding_size)
        self.turn_embeddings= nn.Embedding(config.max_turn_embeddings, embedding_size)
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=True)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )
        self.register_buffer(
            "turn_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

    def forward(
        self, input_ids=None, position_ids=None, turn_ids= None, token_type_ids=None, **kwargs #role_ids=None, 
    ):
        # input size:
        #### B*S*L
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if turn_ids is None:
            if hasattr(self, "turn_ids"):
                buffered_turn_ids = self.turn_ids[:, :seq_length]
                buffered_turn_ids_expanded = buffered_turn_ids.expand(input_shape[0], seq_length)
                turn_ids = buffered_turn_ids_expanded
            else:
                turn_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        # if role_ids is None:
        #     if hasattr(self, "role_ids"):
        #         buffered_role_ids = self.role_ids[:, :seq_length]
        #         buffered_role_ids_expanded = buffered_role_ids.expand(input_shape[0], seq_length)
        #         role_ids = buffered_role_ids_expanded
        #     else:
        #         role_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        turn_embeddings = self.turn_embeddings(turn_ids)
        embeddings = inputs_embeds + token_type_embeddings +turn_embeddings #+ role_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# intact
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: FlowBertConfig):
        super().__init__()
        embedding_size = getattr(config, 'embedding_size', config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, **kwargs
    ):
        input_shape = input_ids.size()

        seq_length = input_shape[1]

        position_ids = self.position_ids[:, : seq_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# intact
class SelfAttention(nn.Module):
    def __init__(self, config: FlowBertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.


        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))


        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in ElectraModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer


# intact
class AttentionOutput(nn.Module):
    def __init__(self, config: FlowBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# intact
class SelfAttentionModule(nn.Module):
    def __init__(self, config: FlowBertConfig):
        super().__init__()
        self.self = SelfAttention(config)
        self.output = AttentionOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
        )
        attention_output = self.output(self_outputs, hidden_states)
        # outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # return outputs
        return attention_output


# intact
class IntermediateLayer(nn.Module):
    def __init__(self, config: FlowBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
           
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
            
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# intact
class SegmentOutput(nn.Module):
    def __init__(self, config: FlowBertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# intact
class SelfAttentionLayer(nn.Module):
    def __init__(self, config: FlowBertConfig):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SelfAttentionModule(config)
        self.intermediate = IntermediateLayer(config)
        self.output = SegmentOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,

    ):
        attention_output = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask
        )
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SegmentPooler(nn.Module):
    def __init__(self, config: FlowBertConfig):
        super().__init__()
        self.segment_size = config.segment_size
        self.hidden_size = config.hidden_size
        self.dense= nn.Linear(self.hidden_size, self.hidden_size)
        
    def get_shape(self, config: FlowBertConfig):
        if config is not None:
            segment_size = config.segment_size
            return segment_size
        else:
            return self.segment_size
        
    def forward(
            self,
            utterance_states: torch.Tensor,
            utterance_mask: torch.Tensor,
            config=None
        ):
        """
        Pool segments

        Params
        ------
            - `utterance_states`: (B*S, L, d)
            - `utterance_mask`: (B*S, 1, 1, L)

        Return
        ------
            - `dialogue_states`: (B*S, d)
            - `dialogue_mask`: (B*S, 1)
        
        Notation
        --------
        B: batch size
        S: segments number
        d: hidden size
        L: size of segment
        """

        # (B*S, L, d)
        segment_size = self.get_shape(config)

        # (B, 1, S*L)
        hidden_size = self.hidden_size

        # (B*S, L, 1)
        utterance_mask = utterance_mask.view(-1, segment_size, 1)
        utterance_mask_exp = torch.exp(utterance_mask)

        # (B*S, d)
        dialogue_states = torch.sum(utterance_mask_exp * utterance_states, dim=1) / (1e-6 + utterance_mask_exp.sum(dim=1) )
        dialogue_mask = (1.0 - utterance_mask_exp.sum(dim=1)) * (-10000.0)

        # (B*S, d, 1)
        dialogue_states = dialogue_states.unsqueeze(2) 

        # (B*S, L, 1)
        scores = torch.bmm(utterance_states, dialogue_states) / math.sqrt(hidden_size)
        scores += utterance_mask
        scores = torch.softmax(scores, dim=1)

        # (B*S, d)
        dialogue_states = torch.sum(utterance_states * scores, dim=1)
        dialogue_states = self.dense(dialogue_states)

        return dialogue_states, dialogue_mask


class SegmentUpdation(nn.Module):
    def __init__(self, config: FlowBertConfig):
        super().__init__()
        self.hidden_size = config.hidden_size

    def forward(
        self,
        dialogue_states: torch.Tensor,
        utterance_states: torch.Tensor,
        utterance_mask: torch.Tensor
    ):
        """
        Update hidden states of each token in dialogue.

        dialogue_states: (B, S, d)
        utterance_states: (B*S, L, d)
        utterance_mask: (B*S, 1, 1, L)

        where
        B: batch size
        S: segments number
        d: hidden size
        L: size of segment
        """


        # (B*S, d, 1)
        dialogue_states = dialogue_states.unsqueeze(2) 

        # (B*S, L, 1), scores for tokens 
        scores = torch.bmm(utterance_states, dialogue_states) / math.sqrt(self.hidden_size)
        scores += utterance_mask.squeeze(1).squeeze(1).unsqueeze(2)
        scores = torch.softmax(scores, dim=1)

        # (B*S, L, d) + (B*S, L, 1) x (B*S, 1, d)
        hidden_states = utterance_states +  scores * dialogue_states.view(-1, 1, self.hidden_size)
        
        # (B*S, L, d)
        return hidden_states


# instead of SelfAttentionModule (BertAttention)
class FlowAttention(nn.Module):
    def __init__(self, config: FlowBertConfig):
        super().__init__()
        self.segment_count = config.seq_length // config.segment_size
        assert config.seq_length % config.segment_size == 0, f"seq_length must be multiplication of segment size: {config.seq_length}/{config.segment_size}"
        self.segment_size = config.segment_size
        self.hidden_size = config.hidden_size

        self.self = SelfAttention(config)       # intact
        self.bpooler = SegmentPooler(config)
        self.flow = SegmentUpdation(config)
        self.output = AttentionOutput(config)   # intact

    def get_shape(self, config: FlowBertConfig):
        if config is not None:
            segment_count = config.seq_length // config.segment_size
            assert config.seq_length % config.segment_size == 0, f"seq_length must be multiplication of segment size: {config.seq_length}/{config.segment_size}"
            segment_size = config.segment_size
            return segment_count, segment_size
        else:
            return self.segment_count, self.segment_size
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        config=None,
    ):
        # hidden_states: (B, S*L, d)
        bsz = hidden_states.size(0)
        utterance_count, seq_length = self.get_shape(config)
        hidden_size = self.hidden_size

        # utterance-level attention
        # (B*S, L, d)
        utterance_states = hidden_states.view(bsz * utterance_count, seq_length, hidden_size)
        # (B*S, 1, 1, L)
        utterance_mask = attention_mask.view(bsz * utterance_count, 1, 1, seq_length)
        
        # (B*S, L, d)
        utterance_states = self.self(
            utterance_states,
            attention_mask=utterance_mask   # needed to be replaced with 2D mask
        )

        # using pooling method to get segment repr
        # (B*S, d) and (B*S, 1)
        dialogue_states, dialogue_mask = self.bpooler(utterance_states, attention_mask, config)
        # (B, S, d)
        dialogue_states = dialogue_states.view(bsz, utterance_count, hidden_size)
        # (B, 1, 1, S)
        dialogue_mask = dialogue_mask.view(bsz, 1, 1, utterance_count)
        
        # utterances iteraction 
        dialogue_states = self.self(
            dialogue_states,
            attention_mask=dialogue_mask
        )
        # update the token hidden states with correspondng segment states
        # (B*S, L, d)
        utterance_states = self.flow(
            dialogue_states,
            utterance_states,
            utterance_mask
        )

        # (B, S*L, d)
        utterance_states = utterance_states.view(bsz, utterance_count * seq_length, hidden_size)

        flowed_states = self.output(utterance_states, hidden_states)
        return flowed_states


# instead of SelfAttentionLayer (BertLayer)
class AttentionFlowLayer(nn.Module):
    def __init__(self, config: FlowBertConfig):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = FlowAttention(config)
        self.intermediate = IntermediateLayer(config)
        self.output = SegmentOutput(config)


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        config = None,
    ):
        attention_output = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            config= config
        ) #B*S*L*d
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        
        return layer_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# instead of ModelOutput
@dataclass
class FlowEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    cls_state: torch.FloatTensor = None


# adopt AttentionFlowLayer to hf format
class FlowBERTEncoder(nn.Module):
    def __init__(self, config: FlowBertConfig):
        super().__init__()

        self.config = copy.deepcopy(config)

        # list of segment sizes for each transformer layer
        segment_list = config.segment_size

        if isinstance(config.num_hidden_layers, int):
            layers = []
            for i in range(config.num_hidden_layers):
                if i < len(segment_list) and segment_list[i] > 0:
                    blsz = segment_list[i]
                    config = copy.deepcopy(config)
                    config.segment_size = blsz
                    layers.append(AttentionFlowLayer(config))
                else:
                    config.blsz = -1
                    layers.append(SelfAttentionLayer(config))
        else:
            raise ValueError(f"{self.config.num_hidden_layers} are not supported for FlowBERT")
        
        self.layer = nn.ModuleList(layers)

    def forward(
        self,
        hidden_state,
        attention_mask=None,
    ):
        for layer_module in self.layer:
            hidden_state = layer_module(
                hidden_state,
                attention_mask=attention_mask
            )
            
        outs = FlowEncoderOutput(hidden_state)
        return outs


class FlowBERTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FlowBertConfig
    base_model_prefix = "bert" # we set this to load bert embeddings
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
                module.bias.data.add_(0.1)

    def get_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        
        
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        if attention_mask.ndim==2:
            attention_mask=attention_mask[:, None, None, :] # B*1*1*L
        elif attention_mask.ndim==3:
            attention_mask = torch.sum(attention_mask, dim=2, keepdim=True)
            attention_mask=attention_mask.unsqueeze(1) # B*1*m*L
        else:
            raise ValueError(f"ndim({attention_mask.ndim}) of attention mask is not supported")
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0
        return attention_mask
    

class FlowBERTModel(FlowBERTPreTrainedModel):
    def __init__(self, config: FlowBertConfig):
        super().__init__(config)
        if config.jdembedding:
            self.embeddings = DialEmbeddings(config)
        else:
            self.embeddings = BertEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)
        
        self.encoder = FlowBERTEncoder(config) #SpanConvEncoder(config)
        self.config = config
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids= None,
        turn_ids = None,
        role_ids = None,
    ):
        # input_ids, attention_mask, token_type_ids: B*SL
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = self.get_attention_mask(attention_mask)
        hidden_state = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            role_ids = role_ids,
            turn_ids =  turn_ids,
            position_ids = position_ids
        ) #B*SL*d
        if hasattr(self, "embeddings_project"):
            hidden_state = self.embeddings_project(hidden_state)

        outputs = self.encoder(
            hidden_state,
            attention_mask=attention_mask,
        )

        return outputs


class JDTokenizerCompatible(BertTokenizerFast):
    vocab_files_names = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}
    def __init__(
        self,
        vocab_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        eou_token="[EOU]",
        sys_token="[SYS]",
        user_token="[USER]",
        url_token="JDHTTP",
        phone_token="JDPHONE",
        utterance_token="[UMASK]",
        span_token="[SMASK]",
        role_token="[RMASK]",

        **kwargs
    ) -> None:
        do_lower_case = kwargs.pop("do_lower_case", True)
        do_basic_tokenize = kwargs.pop("do_basic_tokenize", True)
        never_split = kwargs.pop("never_split", None)
        tokenize_chinese_chars = kwargs.pop("tokenize_chinese_chars", True)
        strip_accents = kwargs.pop("strip_accents", None)
        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split= never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars= tokenize_chinese_chars,
            strip_accents= strip_accents,
            **kwargs
        )
        self.add_special_tokens({"additional_special_tokens":[eou_token,user_token, sys_token, utterance_token, span_token, role_token,url_token,phone_token]})

        self.user_token=user_token
        self.sys_token=sys_token
        self.span_token = span_token
        self.utterance_token= utterance_token
        self.role_token = role_token
        self.eou_token=  eou_token
        
        self.wordpieces_prefix = "##"
        self.simpleMode=False

    def tokenize(self, text, **kwargs):
        # only works for simpleMode
        # if text.startswith("##"):
        #     print("t-tune",text)
        if self.simpleMode==False or kwargs['is_split_into_words']==False:
            token_list= super().tokenize(text, *kwargs)
            token_list=filter(lambda tok : tok!= SPIECE_UNDERLINE, token_list)
        else:
            token_list = text.split(" ")
        # if text.startswith("##"):
        #     print("t-tune2",token_list)
        return list(token_list)


def tokenize_example(tokenizer: JDTokenizerCompatible, example, max_length=None, pre_tokenized=False, reversed_turn=True, add_turn=False, add_role=False):
    """Tokenise batched examples using a pre-trained RoBERTa tokeniser."""
    if "sentence" in example:
        if pre_tokenized == True and isinstance(tokenizer, PreTrainedTokenizerFast):
            sentences = list(
                map(lambda toks: tokenizer.convert_tokens_to_string(toks), example['sentence']))
            pre_tokenized = False
        else:
            sentences = list(example['sentence'])
        result = tokenizer(sentences,
                           max_length=max_length, padding="max_length", truncation="longest_first",
                           is_split_into_words=pre_tokenized
                           )

    elif "sentence1" in example and "sentence2" in example:
        if pre_tokenized == True and isinstance(tokenizer, PreTrainedTokenizerFast):
            sentences1 = list(
                map(lambda toks: tokenizer.convert_tokens_to_string(toks), example['sentence1']))
            sentences2 = list(
                map(lambda toks: tokenizer.convert_tokens_to_string(toks), example['sentence2']))
            pre_tokenized = False
        else:
            sentences1 = list(example['sentence1'])
            sentences2 = list(example['sentence2'])
        result = tokenizer(sentences1, sentences2,
                           max_length=max_length, padding="max_length", truncation="longest_first",
                           is_split_into_words=pre_tokenized
                           )
    else:
        raise ValueError(f"not suported fields: {example.keys()}")

    if "label" in example:
        result["label"] = example["label"]

    batched_input_ids = result["input_ids"]
    eou_id = tokenizer.convert_tokens_to_ids(
        tokenizer.eou_token) if hasattr(tokenizer, 'eou_token') else None
    pad_id = tokenizer.pad_token_id

    if add_role or add_turn:
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
    if add_role:
        role1_id = tokenizer.convert_tokens_to_ids(
            tokenizer.user_token) if hasattr(tokenizer, 'usr_token') else None
        role2_id = tokenizer.convert_tokens_to_ids(
            tokenizer.sys_token) if hasattr(tokenizer, 'sys_token') else None
        role_mapper = {
            role1_id: 1,
            role2_id: 2
        }

    def build_role_ids(input_ids):
        role_ids = []
        for t_id in input_ids:
            if t_id == cls_id:
                role_ids.append(0)
            elif t_id == role1_id or t_id == role2_id:
                role_ids.append(role_mapper[t_id])
            elif t_id == pad_id:
                break
            else:
                role_ids.append(role_ids[-1])
        role_ids += [0] * (len(input_ids)-len(role_ids))
        return role_ids

    def build_turn_ids(input_ids):
        turn_ids = []
        turn_count = 0
        for t_id in input_ids:
            if t_id == pad_id or t_id == sep_id:
                break
            turn_ids.append(turn_count)
            if t_id == eou_id:
                turn_count += 1
        if reversed_turn:
            max_turn_id = max(turn_ids)
            # pad and cls is 0, the rest is from T,T-1,T-2...,1
            turn_ids = turn_ids[:1] + [max_turn_id -
                                       turn + 1 for turn in turn_ids[1:]]
        turn_ids += [0] * (len(input_ids)-len(turn_ids))
        return turn_ids

    def build_utterance_labels(input_ids, index, label):
        utterance_labels = []
        count = 0
        for t_id in input_ids:
            if t_id == eou_id:
                if index == count:
                    utterance_labels.append(label)
                else:
                    utterance_labels.append(0)
                count += 1
            elif t_id == sep_id or t_id == pad_id:
                break
            else:
                utterance_labels.append(-100)
        utterance_labels += [-100] * (len(input_ids)-len(utterance_labels))
        return utterance_labels

    def assign_utterance_labels(input_ids, label_list):
        utterance_labels = []
        count = 0
        for t_id in input_ids:
            if t_id == eou_id:
                utterance_labels.append(label_list[count])
                count += 1
            elif t_id == pad_id:
                break
            else:
                utterance_labels.append(-100)
        
        assert count == len(label_list), (
            f"""utterance size and label size mis-match: 
            {count} / {len(label_list)} 
            \n {input_ids}\n{label_list}
            \n {tokenizer.convert_ids_to_tokens(input_ids)}"""
        )

        utterance_labels += [-100] * (len(input_ids)-len(utterance_labels))
        return utterance_labels

    if add_turn:
        batched_turn_ids = list(
            map(lambda x: build_turn_ids(x), batched_input_ids))
        result["turn_ids"] = batched_turn_ids
    if add_role:
        batched_role_ids = list(
            map(lambda x: build_role_ids(x), batched_input_ids))
        result["role_ids"] = batched_role_ids
    # print("result", result.keys())
    if 'label' in result and isinstance(result['label'][0], list):
        batched_label_list = result.pop("label")
        kwargs_list = [{"input_ids": z[0],  "label_list":z[1]}
                       for z in zip(batched_input_ids, batched_label_list)]
        result["utterance_label"] = list(
            map(lambda x: assign_utterance_labels(**x),  kwargs_list))
    elif "index" in example:
        batched_label = result.pop("label")
        batched_index = example["index"]
        kwargs_list = [{"input_ids": z[0], "index": z[1], "label":z[2]}
                       for z in zip(batched_input_ids, batched_index, batched_label)]
        result["utterance_label"] = list(
            map(lambda x: build_utterance_labels(**x),  kwargs_list))

    # example.update(result)
    # result = example
    return result