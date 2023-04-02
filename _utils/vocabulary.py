# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import re
import json
import numpy as np
import pandas as pd
from config import configure as cfg


class Vocab(object):
    WARNING_LENGTH = cfg.warning_len
    RECORDS_SAVE_DIR = cfg.model_data
    number_compiler = re.compile(r"[a-zA-Z]", re.S)
    letter_compiler = re.compile(r"[0-9]", re.S)
    symbol_compiler = re.compile(r"[\W_]", re.S)
    char_compiler = re.compile(r"[\u4e00-\u9fa5]", re.S)

    def __init__(self,
                 verbose: bool,
                 dataset: str,
                 file_dir: str):
        self.verbose = verbose
        self.dataset = dataset
        self.tokens_counter = dict()
        self.file_path = os.path.join(file_dir, "{}/train.txt".format(dataset))

    def preprocess(self):
        if self.verbose: print('reading file {} ...'.format(self.file_path))
        assert os.path.exists(self.file_path)

        self.setences = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if self.verbose and not (idx + 1) % self.WARNING_LENGTH:
                    print('Warning, line: {}'.format(idx + 1))
                self.setences.append(line.strip())

    def update_state(self, x, state):
        if not x.__len__():
            state.append(False)
        else:
            state.append(True)
        return state

    def assign_state(self, tokens):

        chars_state = []
        symbols_state = []
        letters_state = []
        numbers_state = []
        for token in tokens:
            chars = self.char_compiler.findall(token)
            symbols = self.symbol_compiler.findall(token)
            letters = self.letter_compiler.findall(token)
            numbers = self.number_compiler.findall(token)
            chars_state = self.update_state(chars, chars_state)
            symbols_state = self.update_state(symbols, symbols_state)
            letters_state = self.update_state(letters, letters_state)
            numbers_state = self.update_state(numbers, numbers_state)

        chars_state = np.array(chars_state)
        symbols_state = np.array(symbols_state)
        letters_state = np.array(letters_state)
        numbers_state = np.array(numbers_state)

        bool_states = np.concatenate([chars_state[:, None],
                                      symbols_state[:, None],
                                      letters_state[:, None],
                                      numbers_state[:, None]], axis=-1)

        index_state = np.max(bool_states.astype(int), axis=-1)
        specials_state = np.ones(shape=index_state.shape) - index_state
        specials_state = specials_state.astype(bool)

        return chars_state, symbols_state, letters_state, numbers_state, specials_state

    def get_vocab(self):
        assert self.dataset in ["doupo", "poetry", "tangshi", "zhihu"]

        vocab_name = "{}_vocabulary.json".format(self.dataset)
        vocab_path = os.path.join(self.RECORDS_SAVE_DIR, vocab_name)
        if os.path.exists(vocab_path):
            f = open(vocab_path, 'r')
            dict = json.load(f)

            print("Done loading {} vocabulary".format(self.dataset))

            return dict['vocab']

        self.preprocess()

        for sentence in self.setences:
            for token in list(sentence):
                if token not in self.tokens_counter.keys():
                    self.tokens_counter.update({token: 1})
                else:
                    self.tokens_counter[token] += 1

        tokens_counter = list(self.tokens_counter.values())
        tokens = list(self.tokens_counter.keys())

        index = sorted(np.arange(tokens_counter.__len__()), key=lambda i: tokens_counter[i],
                       reverse=True)

        tokens = np.array(tokens)[index]

        chars_state, symbols_state, \
        letters_state, numbers_state, specials_state = self.assign_state(tokens)

        char_tokens = tokens[chars_state]
        symbol_tokens = tokens[symbols_state]
        letter_tokens = tokens[letters_state]
        number_tokens = tokens[numbers_state]
        special_tokens = tokens[specials_state]

        sorted_tokens = np.concatenate([symbol_tokens,
                                        char_tokens,
                                        letter_tokens,
                                        number_tokens,
                                        special_tokens],
                                       axis=0).tolist()

        vocab = ["<eos>"]
        vocab.extend(sorted_tokens)
        dict = {'vocab': vocab}
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(dict, f, indent=4)
        print("Done writing {} vocabulary".format(self.dataset))

        return vocab


if __name__ == '__main__':
    vocab = Vocab(
        verbose=True,
        dataset="文本名称",
        file_dir="文本数据目录")
    vocabulary = vocab.get_vocab()
    x = vocabulary.index("<eos>") # 添加文本分隔符
