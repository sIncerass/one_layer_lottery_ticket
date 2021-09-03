# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    PositionalEmbedding,
)

from fairseq.modules.masked_modules import MaskedLayerNorm_select, MaskedLinear, MaskedEmbedding
from fairseq.modules.masked_multihead_attention import MaskedMultiheadAttention
from fairseq.modules import MultiheadAttention
from fairseq.modules.masked_transformer_sentence_encoder_layer import MaskedTransformerSentenceEncoderLayer

from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

import functools

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

torch.set_printoptions(precision=10)
def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, MaskedLinear) or isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, MaskedEmbedding) or isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MaskedMultiheadAttention) or isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)


class MaskedTransformerSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        args,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU

        self.embed_tokens = self.build_embedding(
            self.vocab_size, self.embedding_dim, self.padding_idx, args
        )
        
        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                MaskedLinear(input_dim, output_dim, bias=False, prune_ratio=args.prune_ratio, 
                prune_method=args.prune_method, mask_init=args.mask_init, mask_constant=args.mask_constant,
                init=args.init, nonlinearity=args.activation_fn, scale_fan=args.scale_fan), 
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )
        
        if self.embed_positions is not None:
            self.embed_positions.requires_grad = False

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_transformer_sentence_encoder_layer(
                    args,
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        if args.share_mask != 'none':
            if 'layer' in args.share_mask:
                if args.share_mask == 'layer_mask': 
                    share_tag = 'scores'
                elif args.share_mask == 'layer_weights':
                    share_tag = 'weight'
                else:
                    raise ValueError("Shit share-mask")
                for i in range(num_encoder_layers):
                    if i == 0:
                        param_dict = dict()
                        for module_name, m in self.layers[i].named_modules():
                            for param_name, param in m.named_parameters():
                                if share_tag in param_name:
                                    name = ".".join([module_name, param_name]) if module_name != "" else param_name
                                    param_dict[name] = param
                                    print(name)
                    else:
                        print(param_dict.keys())
                        layer = self.layers[i]
                        for module_name, m in layer.named_modules():
                            for param_name, param in m.named_parameters():
                                name = ".".join([module_name, param_name]) if module_name != "" else param_name
                                if name in param_dict:
                                    # name = ".".join([module_name, param_name]) if module_name != "" else param_name
                                    rsetattr(layer, name, param_dict[name])

        if encoder_normalize_before:
            self.emb_layer_norm = MaskedLayerNorm_select(self.embedding_dim, args.mask_layernorm_type)
        else:
            self.emb_layer_norm = None

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if args.weight_trainable:
            for layer in self.layers:
                for name, param in layer.named_parameters():
                    if 'weight' in name:
                        param.requires_grad = True

            for name, param in self.embed_tokens.named_parameters():
                if 'weight' in name:
                    param.requires_grad = True

        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def build_embedding(self, vocab_size, embedding_dim, padding_idx, args):
        # if args.embed_no_mask:
        # m = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
            # m.weight.requires_grad = False
        # else:
        m = MaskedEmbedding(vocab_size, embedding_dim, padding_idx=padding_idx,
                 prune_ratio=args.prune_ratio, prune_method=args.prune_method,
                 mask_init=args.mask_init, mask_constant=args.mask_constant,
                 init=args.init, scale_fan=args.scale_fan, init_embedding_seperate=args.init_embedding_seperate,
                 dynamic_scaling=args.dynamic_scaling,)
        # if args.init_embedding_seperate:
        #     if args.scale_fan and not args.embed_no_mask:
        #         nn.init.normal_(m.weight, mean=0, std=(embedding_dim * (1-args.prune_ratio)) ** -0.5)
        #     else:
        #         nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        #     nn.init.constant_(m.weight[padding_idx], 0)
        return m

    def build_transformer_sentence_encoder_layer(
        self,
        args,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
    ):
        return MaskedTransformerSentenceEncoderLayer(
            args,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)
        if not self.traceable and not self.tpu and not padding_mask.any():
            padding_mask = None

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.embed_tokens(tokens)

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.embed_positions is not None:
            x = x + self.embed_positions(tokens, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x = x + self.segment_embeddings(segment_labels)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        # print('inner result: ', x[0,0,100].data)
        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
            # print('inner result: ', x[0,0,100].data)
            if not last_state_only:
                inner_states.append(x)

        sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep
