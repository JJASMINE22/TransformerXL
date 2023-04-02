# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import numpy as np
import tensorflow as tf


def char_recursion(chars_list: list):

    chars_list[0] = chars_list[0] + chars_list[1]
    del chars_list[1]
    while chars_list.__len__() > 1:
        char_recursion(chars_list)

    return chars_list[0]

def create_tfrecords(save_dir, dataset: str, part: str,
                     batched_data: np.ndarray, batch_size: int, seq_len: int):
    assert dataset in ["doupo", "poetry", "tangshi", "zhihu"]
    file_name = "{}-part-{}-batch_size-{}.seq_len-{}.tfrecords".format(dataset, part, batch_size, seq_len)
    file_path = os.path.join(save_dir, file_name)
    if os.path.exists(file_path):
        return {"{}_file_path".format(part): file_path,
                "{}_steps".format(part): (batched_data.shape[1] - 1) // seq_len + 1}
    steps = 0
    with tf.io.TFRecordWriter(file_path) as file_writer:
        for i in range(0, batched_data.shape[1] - 1, seq_len):
            cur_seq_len = min(batched_data.shape[1] - 1 - i, seq_len)
            if not (steps + 1) % 500:
                print("processing steps: {}".format(steps))
            for batch_num in range(batch_size):
                source = batched_data[batch_num, i: i + cur_seq_len]
                target = batched_data[batch_num, i + 1:i + 1 + cur_seq_len]
                """
                Var type accepted by tf.train.Feature() is usually List, 
                Batch data is suitable to be stored in a single sequence format
                """
                record_bytes = tf.train.Example(features=tf.train.Features(feature={
                    "source": tf.train.Feature(int64_list=tf.train.Int64List(value=source.tolist())),
                    "target": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=target.tolist()))})).SerializeToString()

                file_writer.write(record_bytes)
            steps += 1

        file_writer.close()

    print("Done writing {}. steps: {}".format(file_name, steps))
    return {"{}_file_path".format(part): file_path,
            "{}_steps".format(part): steps}


def decode_fn(record_bytes):
    return tf.io.parse_single_example(
        # Data
        record_bytes,

        # Schema
        {"source": tf.io.VarLenFeature(dtype=tf.int64),
         "target": tf.io.VarLenFeature(dtype=tf.int64)}
    )


def cosine_decay_with_warmup(global_step,
                             total_steps,
                             warmup_steps,
                             hold_steps,
                             learning_rate_base,
                             warmup_learning_rate,
                             min_learning_rate):
    if any([learning_rate_base, warmup_learning_rate, min_learning_rate]) < 0:
        raise ValueError('all of the learning rates must be greater than 0.')

    if np.logical_or(total_steps < warmup_steps, total_steps < hold_steps):
        raise ValueError('total_steps must be larger or equal to the other steps.')

    if np.logical_or(learning_rate_base < min_learning_rate, warmup_learning_rate < min_learning_rate):
        raise ValueError('learning_rate_base and warmup_learning_rate must be larger or equal to min_learning_rate.')

    if learning_rate_base < warmup_learning_rate:
        raise ValueError('learning_rate_base must be larger or equal to warmup_learning_rate.')

    if global_step < warmup_steps:

        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        return slope * global_step + warmup_learning_rate

    elif warmup_steps <= global_step <= warmup_steps + hold_steps:

        return learning_rate_base

    else:
        return 0.5 * learning_rate_base * (1 + np.cos(np.pi * (global_step - warmup_steps - hold_steps) /
                                                      (total_steps - warmup_steps - hold_steps)))


class WarmUpCosineDecayScheduler:

    def __init__(self,
                 global_step=0,
                 global_step_init=0,
                 global_interval_steps=None,
                 warmup_interval_steps=None,
                 hold_interval_steps=None,
                 learning_rate_base=None,
                 warmup_learning_rate=None,
                 min_learning_rate=None,
                 interval_epoch=[0.05, 0.15, 0.3, 0.5],
                 verbose=None,
                 **kwargs):
        self.global_step = global_step
        self.global_steps_for_interval = global_step_init
        self.global_interval_steps = global_interval_steps
        self.warmup_interval_steps = warmup_interval_steps
        self.hold_interval_steps = hold_interval_steps
        self.learning_rate_base = learning_rate_base
        self.warmup_learning_rate = warmup_learning_rate
        self.min_learning_rate = min_learning_rate
        self.interval_index = 0
        self.interval_epoch = interval_epoch
        self.interval_reset = [self.interval_epoch[0]]
        for i in range(len(self.interval_epoch) - 1):
            self.interval_reset.append(self.interval_epoch[i + 1] - self.interval_epoch[i])
        self.interval_reset.append(1 - self.interval_epoch[-1])
        self.verbose = verbose

    def batch_begin(self):
        if self.global_steps_for_interval in [0] + [int(j * self.global_interval_steps) for j in self.interval_epoch]:
            self.total_steps = int(self.global_interval_steps * self.interval_reset[self.interval_index])
            self.warmup_steps = int(self.warmup_interval_steps * self.interval_reset[self.interval_index])
            self.hold_steps = int(self.hold_interval_steps * self.interval_reset[self.interval_index])
            self.interval_index += 1
            self.global_step = 0

        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      total_steps=self.total_steps,
                                      warmup_steps=self.warmup_steps,
                                      hold_steps=self.hold_steps,
                                      learning_rate_base=self.learning_rate_base,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      min_learning_rate=self.min_learning_rate)

        self.global_step += 1
        self.global_steps_for_interval += 1

        if self.verbose:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_steps_for_interval, lr))

        return lr
