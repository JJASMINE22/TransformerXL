# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import re
import numpy as np
import tensorflow as tf
from _utils.vocabulary import Vocab
from _utils.utils import create_tfrecords, decode_fn
from config import configure as cfg


class Generator(object):
    WARNING_LENGTH = cfg.warning_len
    symbol_compiler = re.compile(r"[\W_]", re.S)
    RECORDS_SAVE_DIR = cfg.model_data

    def __init__(self,
                 verbose: bool,
                 dataset: str,
                 file_dir: str,
                 batch_size: int,
                 sequence_len: int):
        self.verbose = verbose
        self.dataset = dataset
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.vocabulary = Vocab(verbose=verbose,
                                dataset=dataset,
                                file_dir=file_dir).get_vocab()
        self.train_fp = os.path.join(file_dir, "{}/train.txt".format(dataset))
        self.valid_fp = os.path.join(file_dir, "{}/valid.txt".format(dataset))

    def tokenize(self, line, add_eos=False):
        tokens = list(line)
        symbols = self.symbol_compiler.findall(line)

        if tokens.__len__() == symbols.__len__():
            return []

        if add_eos:
            return tokens + ['<eos>']
        else:
            return tokens

    def convert_to_ndarray(self, tokens):

        tokens = [self.vocabulary.index(token) for token in tokens]

        return tokens

    def encode(self, file_path, ordered: bool = True):
        if self.verbose: print('encoding file {} ...'.format(file_path))
        assert os.path.exists(file_path)

        encoded_tokens = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if self.verbose and not (idx + 1) % self.WARNING_LENGTH:
                    print('Warning, line: {}'.format(idx + 1))
                tokens = self.tokenize(line.strip(), add_eos=True)

                if not tokens.__len__():
                    continue

                encoded_tokens.append(self.convert_to_ndarray(tokens))

        if ordered:
            encoded_tokens = np.concatenate(encoded_tokens)

        return encoded_tokens

    def batchify(self, tokens):

        steps = tokens.__len__() // self.batch_size
        tokens = tokens[: self.batch_size * steps]
        tokens = np.reshape(tokens, newshape=[self.batch_size, steps])

        return tokens

    def generate_tfrecords(self):

        train_tokens = self.encode(self.train_fp)
        train_tokens = self.batchify(train_tokens)

        train_info = create_tfrecords(save_dir=self.RECORDS_SAVE_DIR,
                                      dataset=self.dataset, part="train",
                                      batched_data=train_tokens,
                                      batch_size=self.batch_size,
                                      seq_len=self.sequence_len)

        if self.valid_fp is not None:
            valid_tokens = self.encode(self.valid_fp)
            valid_tokens = self.batchify(valid_tokens)

            valid_info = create_tfrecords(save_dir=self.RECORDS_SAVE_DIR,
                                          dataset=self.dataset, part="valid",
                                          batched_data=valid_tokens,
                                          batch_size=self.batch_size,
                                          seq_len=self.sequence_len)

            return [train_info, valid_info]

        return [train_info, None]

    def assign_gathered(self, values, indices):

        shape = [self.batch_size, tf.shape(indices)[0] // self.batch_size]

        values = tf.scatter_nd(indices=indices,
                               updates=values,
                               shape=shape).numpy()

        return values
