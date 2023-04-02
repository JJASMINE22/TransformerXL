# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import numpy as np
import tensorflow as tf
from custom.utils import rel_shift, cache_memory
from custom import *


class PositionalEncoding(layers.Layer):
    def __init__(self,
                 max_seq_len: int,
                 embed_size: int,
                 **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

        freq = [1 / (10000 ** (i / embed_size)) for i in range(0, embed_size, 2)]
        seq_pos = np.arange(max_seq_len).astype('float32')

        grids = tf.meshgrid(seq_pos, freq, indexing="ij")
        grids = tf.stack(grids, axis=-1)
        grids = tf.reduce_prod(grids, axis=-1)

        values = tf.concat([tf.sin(grids), tf.cos(grids)], axis=-1)
        values = values[tf.newaxis, ...]

        self.values = tf.Variable(values, trainable=False,
                                  name='pos_emb')

    def call(self, inputs, *args, **kwargs):
        seq_len = tf.shape(inputs)[1]
        x = inputs + self.values[:, :seq_len]

        return x


class MaskAdaptiveEmbedding(layers.Layer):
    def __init__(self,
                 d_proj: int,
                 d_embed: int,
                 n_token: int,
                 cutoffs: list,
                 initializer: str,
                 proj_initializer: str,
                 divide_rate: float = 1.,
                 **kwargs):
        super(MaskAdaptiveEmbedding, self).__init__(**kwargs)
        if cutoffs.__len__(): assert divide_rate >= 1

        self.d_embed = d_embed
        self.d_proj = d_proj
        self.divide_rate = divide_rate
        initializer = initializers.get(initializer)
        proj_initializer = initializers.get(proj_initializer)

        if not cutoffs.__len__():
            self.embeddings = tf.Variable(initial_value=initializer(shape=(n_token, d_embed)),
                                          name='embeddings')
            if d_embed != d_proj:
                self.proj_w = tf.Variable(initial_value=proj_initializer(shape=(d_embed, d_proj)),
                                          name='projs_w')
            else:
                self.proj_w = None

        else:
            j = 0
            self.embeddings = []
            self.proj_w = []
            self.cur_d_embed = []
            self.cutoffs = [0] + cutoffs + [n_token]
            for i in range(self.cutoffs.__len__() - 1):
                l_idx, r_idx = self.cutoffs[i], self.cutoffs[i + 1]
                cur_d_embed = d_embed // (divide_rate ** i)
                self.cur_d_embed.append(cur_d_embed)
                self.embeddings.append(tf.Variable(initial_value=initializer(shape=(r_idx - l_idx, cur_d_embed)),
                                                   name='embeddings{:d}'.format(i)))
                if cur_d_embed != d_proj:
                    self.proj_w.append(tf.Variable(initial_value=proj_initializer(shape=(cur_d_embed, d_proj)),
                                                   name='proj{:d}'.format(j)))
                    j += 1

    def call(self, inputs, *args, **kwargs):

        emb_scale = tf.sqrt(float(self.d_proj))
        if not hasattr(self, "cutoffs"):
            feats = tf.gather(self.embeddings, inputs)
            if self.d_embed != self.d_proj:
                feats = tf.matmul(feats, self.proj_w)
        else:
            size = tf.shape(inputs)
            # feats = tf.zeros(shape=(size[0], size[1], self.d_proj))
            feats = []
            j = 0
            for i in range(self.cutoffs.__len__() - 1):
                l_idx, r_idx = self.cutoffs[i], self.cutoffs[i + 1]
                bool_mask = tf.logical_and(tf.greater_equal(inputs, l_idx),
                                           tf.less(inputs, r_idx))
                masked_inputs = tf.boolean_mask(inputs, bool_mask)
                x = tf.gather(self.embeddings[i], masked_inputs)
                if self.cur_d_embed[i] != self.d_proj:
                    x = tf.matmul(x, self.proj_w[j])
                    j += 1
                index_mask = tf.where(bool_mask)
                feats.append(tf.scatter_nd(index_mask, x,
                                           shape=tf.cast((size[0], size[1], self.d_proj),
                                                         dtype=tf.int64)))
                # feats = feats + tf.scatter_nd(index_mask, x, shape=tf.cast(tf.shape(feats), dtype=tf.int64))
            feats = tf.add_n(feats)

        return feats * emb_scale


class MaskAdaptiveLogSoftmax(layers.Layer):
    def __init__(self,
                 d_proj: int,
                 d_embed: int,
                 n_token: int,
                 cutoffs: list,
                 tie_projs: list,
                 proj_initializer: str,
                 divide_rate: float = 1.,
                 **kwargs):
        super(MaskAdaptiveLogSoftmax, self).__init__(**kwargs)
        if cutoffs.__len__(): assert divide_rate >= 1

        self.d_proj = d_proj
        self.d_embed = d_embed
        self.tie_projs = tie_projs
        self.divide_rate = divide_rate
        proj_initializer = initializers.get(proj_initializer)

        if not cutoffs.__len__():
            self.bias = tf.Variable(initial_value=tf.zeros(shape=(n_token,)),
                                    name='bias')
        else:
            self.cur_bias = []
            self.cur_proj = []
            self.cutoffs = [0] + cutoffs + [n_token]
            for i in range(self.cutoffs.__len__() - 1):
                l_idx, r_idx = self.cutoffs[i], self.cutoffs[i + 1]
                cur_d_embed = d_embed // (divide_rate ** i)
                self.cur_bias.append(tf.Variable(initial_value=tf.zeros(shape=(r_idx - l_idx,)),
                                                 name='cur_bias{:d}'.format(i)))

                if not tie_projs[i]:
                    if cur_d_embed != d_proj:
                        self.cur_proj.append(tf.Variable(initial_value=proj_initializer(shape=(cur_d_embed, d_proj)),
                                                         name='cur_w{:d}'.format(i)))

    def call(self, x, inputs=None, params=None, *args, **kwargs):

        def assign_logits(x, weight, bias, proj=None):

            if proj is not None:
                x = tf.matmul(x, proj, transpose_b=True)
            x = tf.matmul(x, weight, transpose_b=True)
            x = x + bias

            return x

        input_size = tf.shape(inputs)
        params_w, params_proj = params

        if not hasattr(self, 'cutoffs'):
            outputs = assign_logits(x, params_w, self.bias, params_proj)
            outputs = tf.nn.log_softmax(outputs, axis=-1)

            return outputs
        else:
            j, k = 0, 0
            outputs = []
            for i in range(self.cutoffs.__len__() - 1):
                l_idx, r_idx = self.cutoffs[i], self.cutoffs[i + 1]
                output = tf.zeros(shape=(input_size[0], input_size[1],
                                         r_idx - l_idx))
                bool_mask = tf.logical_and(tf.greater_equal(inputs, l_idx),
                                           tf.less(inputs, r_idx))
                index_mask = tf.where(bool_mask)

                cur_d_embed = self.d_embed // (self.divide_rate ** i)

                cur_w = params_w[i]
                cur_b = self.cur_bias[i]

                if self.tie_projs[i]:
                    if self.d_proj != cur_d_embed:
                        cur_proj = params_proj[j]
                        j += 1
                    else:
                        cur_proj = None
                else:
                    if self.d_proj != cur_d_embed:
                        cur_proj = self.cur_proj[k]
                        k += 1
                        j += 1
                    else:
                        cur_proj = None

                logits = assign_logits(x, cur_w, cur_b, cur_proj)
                masked_logits = tf.boolean_mask(logits, bool_mask)
                outputs.append(tf.scatter_nd(index_mask, masked_logits,
                                             shape=tf.cast(tf.shape(output), dtype=tf.int64)))
            outputs = tf.concat(outputs, axis=-1)
            outputs = tf.nn.log_softmax(outputs, axis=-1)

            return outputs


class FeedForwardLayer(layers.Layer):
    def __init__(self,
                 embed_size: int,
                 drop_rate: float = .3,
                 use_bias: bool = False,
                 activation: str = 'relu',
                 kernel_initializer='random_normal',
                 kernel_regularizer='l2',
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 **kwargs):
        super(FeedForwardLayer, self).__init__(**kwargs)

        kernel_initializer = initializers.get(kernel_initializer)
        kernel_regularizer = regularizers.get(kernel_regularizer)
        bias_initializer = initializers.get(bias_initializer)
        bias_regularizer = regularizers.get(bias_regularizer)

        self.init_linear = layers.Dense(units=embed_size * 4,
                                        use_bias=use_bias,
                                        activation=activation,
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=kernel_regularizer,
                                        bias_initializer=bias_initializer,
                                        bias_regularizer=bias_regularizer)
        self.final_linear = layers.Dense(units=embed_size,
                                         use_bias=use_bias,
                                         kernel_initializer=kernel_initializer,
                                         kernel_regularizer=kernel_regularizer,
                                         bias_initializer=bias_initializer,
                                         bias_regularizer=bias_regularizer)

        self.init_dropout = layers.Dropout(rate=drop_rate)
        self.final_dropout = layers.Dropout(rate=drop_rate)

        self.layer_norm = layers.LayerNormalization()

    def call(self, inputs, *args, **kwargs):
        x = self.init_linear(inputs)
        x = self.init_dropout(x)

        x = self.final_linear(x)
        x = self.final_dropout(x)

        x = x + inputs
        x = self.layer_norm(x)

        return x


class rel_multihead_attn(layers.Layer):
    def __init__(self,
                 n_head: int,
                 d_head: int,
                 embed_size: int,
                 use_bias: bool = False,
                 drop_rate: float = .3,
                 attn_drop_rate: float = .3,
                 kernel_initializer='random_normal',
                 kernel_regularizer='l2',
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 **kwargs):
        super(rel_multihead_attn, self).__init__(**kwargs)
        self.n_head = n_head
        self.d_head = d_head
        self.embed_size = embed_size
        kernel_initializer = initializers.get(kernel_initializer)
        kernel_regularizer = regularizers.get(kernel_regularizer)
        bias_initializer = initializers.get(bias_initializer)
        bias_regularizer = regularizers.get(bias_regularizer)

        self.linear_q = layers.Dense(units=n_head * d_head,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     bias_initializer=bias_initializer,
                                     bias_regularizer=bias_regularizer)

        self.linear_k = layers.Dense(units=n_head * d_head,
                                     use_bias=use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_initializer=bias_initializer,
                                     bias_regularizer=bias_regularizer)

        self.linear_v = layers.Dense(units=n_head * d_head,
                                     use_bias=use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_initializer=bias_initializer,
                                     bias_regularizer=bias_regularizer)

        self.linear_r = layers.Dense(units=n_head * d_head,
                                     use_bias=use_bias,
                                     kernel_initializer=self.kernel_initializer,
                                     kernel_regularizer=self.kernel_regularizer,
                                     bias_initializer=bias_initializer,
                                     bias_regularizer=bias_regularizer)

        self.linear = layers.Dense(units=embed_size,
                                   use_bias=use_bias,
                                   kernel_initializer=self.kernel_initializer,
                                   kernel_regularizer=self.kernel_regularizer,
                                   bias_initializer=bias_initializer,
                                   bias_regularizer=bias_regularizer)

        self.layer_norm = layers.LayerNormalization()

        self.dropout = layers.Dropout(rate=drop_rate)
        self.attn_dropout = layers.Dropout(rate=attn_drop_rate)

    def call(self, inputs: list, member=None, *args, **kwargs):
        """
        In addition to member and r_head, other contents are consistent with the MultiHead Attention
        Use .enisum() to complete the linear operation of matrix
        """
        w, r, r_w_bias, r_r_bias, attn_mask = inputs
        batch_size = tf.shape(w)[0]
        q_len = tf.shape(w)[1]
        r_len = tf.shape(r)[0]

        cat = tf.concat([member, w]) \
            if member is not None and member.shape.__len__() > 1 \
            else w

        q = self.linear_q(cat)
        k = self.linear_k(cat)
        v = self.linear_v(cat)
        r = self.linear_r(r)

        q = q[:, -q_len:]

        q = tf.reshape(q, shape=[batch_size, -1, self.n_head, self.d_head])
        k = tf.reshape(k, shape=[batch_size, -1, self.n_head, self.d_head])
        v = tf.reshape(v, shape=[batch_size, -1, self.n_head, self.d_head])
        r = tf.reshape(r, shape=[r_len, self.n_head, self.d_head])

        rw_q = q + r_w_bias
        rr_q = q + r_r_bias

        AC = tf.einsum("bind, bjnd->bijn", rw_q, k)
        BD = tf.einsum("bind, jnd->bijn", rr_q, r)
        BD = rel_shift(BD)

        attention = (AC + BD) / tf.sqrt(float(self.embed_size))
        attn_mask = attn_mask[tf.newaxis, ..., tf.newaxis]
        attention = attention * (1 - attn_mask) - 1e30 * attn_mask

        attn_score = tf.nn.softmax(attention, axis=-2)
        attn_score = self.attn_dropout(attn_score)

        attention = tf.einsum("bijn, bjnd->bind", attn_score, v)
        attention = tf.reshape(attention, shape=[-1, q_len, self.d_head * self.n_head])

        attention = self.linear(attention)
        attention = self.attn_dropout(attention)

        x = w + attention
        x = self.layer_norm(x)

        return x


class RelMultiHeadAttention(layers.Layer):
    def __init__(self,
                 n_head: int,
                 d_head: int,
                 embed_size: int,
                 use_bias: bool = False,
                 drop_rate: float = .3,
                 attn_drop_rate: float = .3,
                 kernel_initializer='random_normal',
                 kernel_regularizer='l2',
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 **kwargs):
        super(RelMultiHeadAttention, self).__init__(**kwargs)
        self.n_head = n_head
        self.embed_size = embed_size
        self.drop_rate = drop_rate
        kernel_initializer = initializers.get(kernel_initializer)
        kernel_regularizer = regularizers.get(kernel_regularizer)
        bias_initializer = initializers.get(bias_initializer)
        bias_regularizer = regularizers.get(bias_regularizer)

        self.linear_q = layers.Dense(units=n_head * d_head,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     bias_initializer=bias_initializer,
                                     bias_regularizer=bias_regularizer)

        self.linear_k = layers.Dense(units=n_head * d_head,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     bias_initializer=bias_initializer,
                                     bias_regularizer=bias_regularizer)

        self.linear_v = layers.Dense(units=n_head * d_head,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     bias_initializer=bias_initializer,
                                     bias_regularizer=bias_regularizer)

        self.linear_r = layers.Dense(units=n_head * d_head,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     bias_initializer=bias_initializer,
                                     bias_regularizer=bias_regularizer)

        self.linear = layers.Dense(units=embed_size,
                                   use_bias=use_bias,
                                   kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   bias_initializer=bias_initializer,
                                   bias_regularizer=bias_regularizer)

        self.layer_norm = layers.LayerNormalization()

        self.dropout = layers.Dropout(rate=drop_rate)
        self.attn_dropout = layers.Dropout(rate=attn_drop_rate)

    def call(self, inputs: list, member=None, *args, **kwargs):
        """
        In addition to member and r_head, other contents are consistent with the MultiHead Attention
        Use .enisum() to complete the linear operation of matrix
        """
        w, r, attn_mask, r_w_bias, r_r_bias = inputs
        batch_size = tf.shape(w)[0]
        q_len = tf.shape(w)[1]
        r_len = tf.shape(r)[0]

        cat = tf.concat([member, w], axis=1) \
            if member is not None and member.shape.__len__() > 1 \
            else w

        q = self.linear_q(cat)
        k = self.linear_k(cat)
        v = self.linear_v(cat)
        r = self.linear_r(r)

        q = q[:, -q_len:]

        r = tf.tile(r, [batch_size, 1, 1])

        q = tf.concat(tf.split(q, num_or_size_splits=self.n_head, axis=-1), axis=0)
        k = tf.concat(tf.split(k, num_or_size_splits=self.n_head, axis=-1), axis=0)
        v = tf.concat(tf.split(v, num_or_size_splits=self.n_head, axis=-1), axis=0)
        r = tf.concat(tf.split(r, num_or_size_splits=self.n_head, axis=-1), axis=0)

        r_w_bias = r_w_bias[:, tf.newaxis]
        r_r_bias = r_r_bias[:, tf.newaxis]
        r_w_bias = tf.repeat(r_w_bias, repeats=batch_size, axis=0)
        r_r_bias = tf.repeat(r_r_bias, repeats=batch_size, axis=0)

        rw_q = q + r_w_bias
        rr_q = q + r_r_bias

        AC = tf.matmul(rw_q, k, transpose_b=True)
        BD = tf.matmul(rr_q, r, transpose_b=True)
        BD = rel_shift(BD)

        attention = (AC + BD) / tf.sqrt(float(self.embed_size))
        attn_mask = attn_mask[tf.newaxis, ...]
        attention = attention * (1 - attn_mask) - 1e30 * attn_mask

        attn_score = tf.nn.softmax(attention, axis=-1)
        attn_score = self.attn_dropout(attn_score)

        attention = tf.matmul(attn_score, v)
        attention = tf.concat(tf.split(attention, num_or_size_splits=self.n_head, axis=0),
                              axis=-1)

        attention = self.linear(attention)
        attention = self.attn_dropout(attention)

        x = w + attention
        x = self.layer_norm(x)

        return x


class DecoderLayer(layers.Layer):
    def __init__(self,
                 n_head: int,
                 d_head: int,
                 embed_size: int,
                 use_bias: bool = False,
                 **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.n_head = n_head
        self.d_head = d_head
        self.attention = RelMultiHeadAttention(n_head=n_head,
                                               d_head=d_head,
                                               embed_size=embed_size,
                                               use_bias=use_bias)
        self.feed_forward = FeedForwardLayer(embed_size=embed_size,
                                             use_bias=use_bias)

    def build(self, input_shape):
        super(DecoderLayer, self).build(input_shape)

        self.r_w_bias = self.add_weight(shape=(self.n_head, self.d_head),
                                        initializer=initializers.zeros(),
                                        trainable=True,
                                        name='r_w_bias')
        self.r_r_bias = self.add_weight(shape=(self.n_head, self.d_head),
                                        initializer=initializers.zeros(),
                                        trainable=True,
                                        name='r_r_bias')
        self.built = True

    def call(self, inputs, member=None, *args, **kwargs):
        inputs += [self.r_w_bias, self.r_r_bias]

        x = self.attention(inputs, member)
        x = self.feed_forward(x)

        return x


class Decoder(layers.Layer):
    def __init__(self,
                 n_head: int,
                 d_head: int,
                 num_layers: int,
                 embed_size: int,
                 member_len: int,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.n_head = n_head
        self.d_head = d_head
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.member_len = member_len
        self.decoder_layers = [DecoderLayer(n_head=n_head,
                                            d_head=d_head,
                                            embed_size=embed_size) for _ in range(num_layers)]

    def call(self, x, pos_embedding=None, attn_mask=None, members=None):
        new_members = []
        for i in range(self.num_layers):
            new_members.append(cache_memory(x, members[i], self.member_len))
            x = self.decoder_layers[i]([x, pos_embedding, attn_mask], members[i])

        return x, new_members
