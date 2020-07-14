# -*- encoding: utf-8 -*-
"""
@file: metrics.py
@time: 2020/6/4 上午10:39
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: 
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from module.core.exception import ParameterError


class WordErrorRate(Metric):

    def __init__(self, name='word_error_rate'):
        super(WordErrorRate, self).__init__(name=name)
        # 初始化参数值。tensor: 0.0。保存叠加错误率
        # 参考：https://tensorflow.google.cn/api_docs/python/tf/keras/metrics/Metric
        self.word_error_rate = self.add_weight(name=self.name,
                                               aggregation=tf.VariableAggregation.SUM,
                                               initializer='zeros')
        # 记录样本次数。
        self.count = 0

    def reset_states(self):
        """
        重置状态值。
        :return:
        """
        self.word_error_rate = self.add_weight(name=self.name, aggregation=tf.VariableAggregation.SUM,
                                               initializer='zeros')
        self.count = 0

    def update_state(self, y_true, y_pred, label_length, black_index=0):
        """
        更新状态。根据输入的标签计算词错误率。
        :param y_true: (tensor, mandatory) 正确标签序列。shape: [sample, label]
        :param y_pred: (tensor, mandatory) 预测标签序列。shape: [sample, pred]
        :param label_length: (tensor, mandatory) 标签序列长度。shape: [sample, 1]
        :param black_index: (int, optional, default=0) 空白字符。用于填充
        :return:
        """
        # self.word_error_rate(tf.ones(1))
        for y_t, y_p, label_len in zip(y_true, y_pred, label_length):
            self.get_word_error_rate(y_t, y_p, label_len[0], black_index)

    def get_word_error_rate(self, y_true, y_pred, label_length, black_index=0):
        """
        词错误率。计算每个样本的词错误率。
        :param y_true: (tensor, mandatory) 样本正确标签。 shape: [label, ]
        :param y_pred: (tensor, mandatory) 样本预测标签。shape: [pred, ]
        :param label_length: (int, mandatory) 标签长度。
        :param black_index: (int, optional, default=0) 空白字符。用于填充
        :return:
        """
        assert len(y_true) >= label_length, \
            ParameterError("The actual label sequence length:{} less than {}".format(len(y_true), label_length))

        y_true = tf.convert_to_tensor(y_true, dtype=tf.int32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.int32)

        # 获取标签字符的序列长度
        y_true = y_true[:label_length]

        # 如果序列长度=0，则设置错误率为 len(y_true) / len(y_true) 即错误率为 100%
        if len(y_pred) == 0:
            error_word_num = len(y_true)
        else:
            # 截断。如果预测字符标签大于正确的标签，则截断。
            if len(y_pred) > len(y_true):
                new_y_p = y_pred[:label_length]
                error_word_num = len(y_pred[label_length:])
            # 填充。如果预测字符标签小于正确的标签，则填充。
            elif len(y_pred) < len(y_true):
                black_count = len(y_true) - len(y_pred)
                new_y_p = tf.concat([y_pred, [black_index] * black_count], axis=0)
                error_word_num = 0
            else:
                new_y_p = y_pred
                error_word_num = 0

            # 统计正确的词数量
            true_count = tf.math.reduce_sum(
                tf.cast(tf.equal(tf.cast(new_y_p, dtype=tf.int32), tf.cast(y_true, dtype=tf.int32)), dtype=tf.int32))
            error_word_num += len(new_y_p) - true_count

        value = tf.squeeze(tf.cast(error_word_num / len(y_true), dtype=tf.float32))
        # print(value)

        self.word_error_rate.assign_add(value)
        self.count += 1

    def result(self):
        """
        返回平均错误率
        :return: (tensor) 平均错误率。
        """
        if self.count == 0:
            value = np.array(1.0)
        else:
            value = self.word_error_rate / self.count
            value = value.numpy()

        return value


def test_module_func():
    y_pred = [np.array([1, 2, 3, 4]),
              np.array([1, 2, 3, 4, 5]),
              np.array([1, 2, 3]),
              np.array([1, 2, 3, 4, 5, 6, 7]),
              np.array([])]

    y_true = np.array([[1, 2, 3, 4, 0, 0, 0],
                       [1, 1, 3, 4, 5, 0, 0],
                       [1, 2, 3, 9, 0, 0, 0],
                       [1, 2, 2, 3, 4, 0, 0],
                       [1, 1, 2, 0, 0, 0, 0]])

    label_length = np.array([[4], [5], [4], [5], [2]])

    wer = WordErrorRate()
    wer(y_true=y_true, y_pred=y_pred, label_length=label_length, black_index=0)
    print("get avg word error rate: {}".format(wer.result()))

    wer.reset_states()
    print("reset state result: ", float(wer.result()))

    wer(y_true=y_true, y_pred=y_pred, label_length=label_length, black_index=0)
    print("reset get avg word error rate: {}".format(wer.result()))

    print("test is ok!")


if __name__ == '__main__':
    test_module_func()
