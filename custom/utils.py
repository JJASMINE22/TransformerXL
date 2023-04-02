import numpy as np
import tensorflow as tf


def cache_memory(curr_feat, prev_mem=None, mem_len: int = 0):
    assert mem_len >= 0
    if tf.shape(prev_mem)[1] == 0:
        mem = curr_feat
    elif mem_len == 0:
        mem = None
    else:
        mem = tf.concat([prev_mem, curr_feat], axis=1)[:, -mem_len:]

    return tf.stop_gradient(mem)


def create_mask(query_len, mems_len, same_len: bool = False):
    attn_mask = tf.ones(shape=(query_len,) * 2)
    attn_mask = tf.linalg.band_part(attn_mask, -1, 0)
    attn_mask = 1 - attn_mask

    if same_len:
        pad_mask = tf.ones(shape=(mems_len,) * 2)
        pad_mask = tf.linalg.band_part(pad_mask, -1, 0)
        dia_mask = tf.linalg.band_part(pad_mask, 0, 0)
        pad_mask = pad_mask - dia_mask
        attn_mask = tf.concat([pad_mask, attn_mask], axis=-1)

        return attn_mask

    pad_mask = tf.zeros(shape=(query_len, mems_len))
    attn_mask = tf.concat([pad_mask, attn_mask], axis=-1)

    return attn_mask


def rel_shift(x):
    x_size = tf.shape(x)

    x = tf.pad(x, paddings=[[0, 0], [0, 0], [1, 0]])
    x = tf.reshape(x, shape=[x_size[0], x_size[2] + 1, x_size[1]])
    x = tf.slice(x, begin=[0, 1, 0], size=[-1, -1, -1])
    x = tf.reshape(x, shape=x_size)

    return x


def build_indices(sequence_len, filter_num, start_index, period):
    sequences = tf.convert_to_tensor(range(sequence_len))
    filters = tf.convert_to_tensor(range(start_index, filter_num, period))
    indices = tf.meshgrid(sequences, filters, indexing="ij")
    indices = tf.stack(indices, axis=-1)
    indices = tf.reshape(indices, [-1, 2])

    return indices


def positional_encoding(seq_pos, freq):
    grids = tf.meshgrid(seq_pos, freq, indexing="ij")
    grids = tf.stack(grids, axis=-1)
    x = tf.reduce_prod(grids, axis=-1)

    sin_x = tf.sin(x[:, 0::2])
    cos_x = tf.cos(x[:, 1::2])
    sin_x = tf.reshape(sin_x, shape=[-1])
    cos_x = tf.reshape(cos_x, shape=[-1])

    pos_emb = tf.zeros(shape=[len(seq_pos), len(freq)])
    sin_indices = build_indices(len(seq_pos), len(freq), 0, 2)
    cos_indices = build_indices(len(seq_pos), len(freq), 1, 2)
    pos_emb += tf.scatter_nd(sin_indices, sin_x, shape=tf.shape(pos_emb))
    pos_emb += tf.scatter_nd(cos_indices, cos_x, shape=tf.shape(pos_emb))

    pos_emb = pos_emb[None]

    return pos_emb
