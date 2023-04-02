# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import numpy as np
import tensorflow as tf
from net import layers, models
from custom.CustomLayers import (PositionalEncoding,
                                 MaskAdaptiveEmbedding,
                                 MaskAdaptiveLogSoftmax,
                                 Decoder)
from custom.utils import create_mask, positional_encoding


class TransformerXLModel(models.Model):
    def __init__(self,
                 n_head: int,
                 d_head: int,
                 d_embed: int,
                 n_token: int,
                 cutoffs: list,
                 tie_projs: list,
                 embed_size: int,
                 divide_rate: int,
                 num_layers: int,
                 member_len: int,
                 **kwargs):
        super(TransformerXLModel, self).__init__(**kwargs)

        self.embed_size = embed_size
        self.num_layers = num_layers

        self.embedding = MaskAdaptiveEmbedding(d_proj=embed_size,
                                               d_embed=d_embed,
                                               n_token=n_token,
                                               cutoffs=cutoffs,
                                               initializer="random_normal",
                                               proj_initializer="random_normal",
                                               divide_rate=divide_rate)

        self.decoder = Decoder(n_head=n_head,
                               d_head=d_head,
                               num_layers=num_layers,
                               embed_size=embed_size,
                               member_len=member_len)

        self.mask_adaptive_logsoftmax = MaskAdaptiveLogSoftmax(d_proj=embed_size,
                                                               d_embed=d_embed,
                                                               n_token=n_token,
                                                               cutoffs=cutoffs,
                                                               tie_projs=tie_projs,
                                                               proj_initializer="random_normal",
                                                               divide_rate=divide_rate)

        self.emb_dropout = layers.Dropout(rate=.3)
        self.pos_emb_dropout = layers.Dropout(rate=.3)

    def call(self, inputs, training=None, mask=None):
        input, members = inputs
        query_len = tf.shape(input)[1]
        mems_len = tf.shape(members[0])[1] if members is not None else 0
        if tf.shape(members)[1] == 0:
            members = [None, ] * self.num_layers
        else:
            members = members

        attn_mask = create_mask(query_len, mems_len)

        emb = self.embedding(input)

        freq = [1 / (10000 ** (i / self.embed_size)) for i in range(self.embed_size)]
        freq = tf.convert_to_tensor(freq, dtype=tf.float32)
        seq_pos = tf.cast(range(mems_len + query_len - 1, -1, -1),
                          dtype=tf.float32)

        pos_emb = positional_encoding(seq_pos, freq)

        x = self.emb_dropout(emb)
        pos_emb = self.pos_emb_dropout(pos_emb)

        x, members = self.decoder(x, pos_embedding=pos_emb,
                                  attn_mask=attn_mask, members=members)

        logits = self.mask_adaptive_logsoftmax(x, inputs=input, params=[self.embedding.embeddings,
                                                                        self.embedding.proj_w])

        return logits, members
