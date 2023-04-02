# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import tensorflow as tf
from custom import losses

class NLLLoss(losses.Loss):
    def __init__(self,
                 **kwargs):
        super(NLLLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):

        total_losses = y_pred * y_true
        total_losses = - total_losses
        total_losses = tf.reduce_sum(total_losses, axis=-1)

        if self.reduction == tf.keras.losses.Reduction.NONE:
            return total_losses
        elif self.reduction == tf.keras.losses.Reduction.SUM:
            return tf.reduce_sum(total_losses)

        return tf.reduce_mean(total_losses)
