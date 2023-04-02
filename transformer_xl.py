# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import numpy as np
import tensorflow as tf
from net.network import TransformerXLModel
from custom.CustomLosses import NLLLoss


class TransformerXL:
    def __init__(self,
                 n_head: int,
                 d_head: int,
                 d_embed: int,
                 n_token: int,
                 cutoffs: list,
                 tie_projs: list,
                 batch_size: int,
                 embed_size: int,
                 divide_rate: int,
                 num_layers: int,
                 member_len: int,
                 learning_rate: float):
        self.model = TransformerXLModel(n_head=n_head,
                                        d_head=d_head,
                                        d_embed=d_embed,
                                        n_token=n_token,
                                        cutoffs=cutoffs,
                                        tie_projs=tie_projs,
                                        embed_size=embed_size,
                                        divide_rate=divide_rate,
                                        num_layers=num_layers,
                                        member_len=member_len)

        self.loss_func = NLLLoss(reduction=tf.keras.losses.Reduction.AUTO)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.train_loss = tf.keras.metrics.Mean()
        self.valid_loss = tf.keras.metrics.Mean()

        self.train_acc = tf.keras.metrics.Accuracy()
        self.valid_acc = tf.keras.metrics.Accuracy()

        self.train_f1_score = tf.keras.metrics.Mean()
        self.valid_f1_score = tf.keras.metrics.Mean()

        self.member_len = member_len
        self.n_token = n_token
        self.members = [np.random.randn(batch_size, 0, embed_size)
                        for i in range(batch_size)]

    # @tf.function
    def train(self, sources, targets):
        targets = tf.one_hot(targets, depth=self.n_token)

        with tf.GradientTape() as tape:
            logits, members = self.model([sources, self.members])
            loss = self.loss_func(targets, logits)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        targets = tf.argmax(targets, axis=-1)
        predictions = tf.argmax(logits, axis=-1)

        self.train_acc(targets, predictions)

        self.members = members

    # @tf.function
    def validate(self, sources, targets):
        targets = tf.one_hot(targets, depth=self.n_token)

        logits, members = self.model([sources, self.members])
        loss = self.loss_func(targets, logits)

        self.valid_loss(loss)
        targets = tf.argmax(targets, axis=-1)
        predictions = tf.argmax(logits, axis=-1)

        self.valid_acc(targets, predictions)

        self.members = members

    def generate_sample(self, sources, batch):
        predictions = []
        members = self.members.copy()
        for i in range(self.member_len):
            logits, members = self.model([sources, members])
            prediction = np.argmax(logits.numpy()[:, -1:], axis=-1)
            predictions.append(prediction)
            sources = prediction
        predictions = np.concatenate(predictions, axis=-1)

        return predictions
