# -*- encoding: utf-8 -*-
"""
@file: transverse.py
@time: 2020/7/3 上午10:58
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: 
"""
import sys
import os
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt

from module.core.utils import template
from module.core.utils import computer_conv_size
from module.core.utils import Writer
from module.core.color import Color
from module.core.exception import FileNotFoundException
from module.core.exception import UnknownError
from module.core.exception import ParameterError


def get_sample_size(dataset, validation_rate=0.1):
    """
    计算训练和验证的总样本大小。
    :param dataset: (tf.data.Dataset, mandatory) 数据集
    :param validation_rate: (float, optional, default=0.1) 划分验证数据集的比例
    :return: (int) 训练样本和验证样本的大小
    """
    if validation_rate > 1 or validation_rate < 0:
        raise ParameterError("validation rate range: 0 ~ 1, actually get:{}".format(validation_rate))

    total_sample_size = len(list(dataset))
    valid_sample_size = int(total_sample_size * validation_rate)
    valid_sample_size = valid_sample_size if valid_sample_size > 0 else 1

    train_sample_size = total_sample_size - valid_sample_size
    return train_sample_size, valid_sample_size


def stem_layer(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False, activation='relu'):
    """ 初始输入层 """
    layer = tf.keras.Sequential([
        Conv1D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name='stem_conv2'),
        BatchNormalization(name='stem_bn_1'),
        Activation(activation, name='stem_activation_1'),
        Conv1D(filters * 2, kernel_size, strides=strides, padding=padding, use_bias=use_bias, name='stem_conv1'),
        BatchNormalization(name='stem_bn_2'),
        Activation(activation, name='stem_activation_2')
    ])
    return layer


def dilated_conv_layer(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False, dilation_rate=2,
                       name='dilated_layer'):
    """ 扩张卷积 """
    layer = tf.keras.Sequential([
        Conv1D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, dilation_rate=dilation_rate,
               name='Dilated_Conv_{}'.format(name)),
        BatchNormalization(name='Dilated_Bn_{}'.format(name))
    ])
    return layer


def packaging_data_generator(gen, shuffle=True, batch_size=8, args=None):
    """
    初始化数据集。
    :param gen: (Generator, mandatory) 数据生成器
    :param shuffle: (bool, optional, default=True) 是否洗牌
    :param batch_size: (int, optional, default=8) 批次样本大小
    :param args: (list, optional, default=None) 需要传入生成器的参数列表
    :return: (tf.data.Dataset) 批次数据集
    """
    output_shapes = (tf.TensorShape([None, None]),
                     tf.TensorShape([None]),
                     tf.TensorShape([None]),
                     tf.TensorShape([None]))
    dataset = tf.data.Dataset.from_generator(gen,
                                             args=args,
                                             output_types=(tf.float32, tf.int32, tf.int32, tf.int32),
                                             output_shapes=output_shapes)

    if shuffle:
        dataset = dataset.shuffle(batch_size * 2).batch(batch_size, drop_remainder=True)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=True)

    # dataset = dataset.cache()
    return dataset


class WidthDilatedConv(tf.keras.layers.Layer):
    """
    横向扩张卷积层。 定义 3 个扩张卷积层 和 1 个普通卷积。

    扩张卷积：
        1. 每个扩张卷积使用相同的 filters、strides、padding。 不同的是 kernel_size 。kernel_size = [2,3,4]。
        2. BN 层。
        3. 最大池化层
    普通卷积：
        1. 普通 (1 * 1) 卷积层。kernel_size=1、strides=1、
        2. BN 层
        3. 最大池化层

    先将每个扩张卷积的结果进行相加。然后再与普通层的结果相加。
    """

    def __init__(self, filters=128, padding='same', use_bias=False, dilation_rate=2, pool_size=3,
                 name='Width_Dilated_conv'):
        """
        :param filters: (int, optional, default=128) 卷积层过滤器数量
        :param padding: (str, optional, default='same') 卷积层是否进行填充， same 或 valid
        :param use_bias: (bool, optional, default=false) 是否使用 bias
        :param dilation_rate: (int, optional, default=2) 扩张卷积层大小。
        :param pool_size: (int, optional, default=3) 最大池化层 pool_size
        """
        super(WidthDilatedConv, self).__init__(name=name)
        self.filters = filters
        self.padding = padding
        self.use_bias = use_bias
        self.dilation_rate = dilation_rate
        self.pool_size = pool_size

        self.__init_layer__()

    def __init_layer__(self):
        self.conv = Conv1D(self.filters, kernel_size=1, strides=1, padding=self.padding, use_bias=self.use_bias,
                           name='Conv1D_{}'.format(self.name))
        self.bn = BatchNormalization(name='Bn_{}'.format(self.name))

        self.dilation_conv1 = dilated_conv_layer(self.filters, kernel_size=2, strides=1, padding=self.padding,
                                                 use_bias=self.use_bias, dilation_rate=self.dilation_rate,
                                                 name='{}_{}'.format(self.name, 1))

        self.dilation_conv2 = dilated_conv_layer(self.filters, kernel_size=3, strides=1, padding=self.padding,
                                                 use_bias=self.use_bias, dilation_rate=self.dilation_rate,
                                                 name='{}_{}'.format(self.name, 2))

        self.dilation_conv3 = dilated_conv_layer(self.filters, kernel_size=4, strides=1, padding=self.padding,
                                                 use_bias=self.use_bias, dilation_rate=self.dilation_rate,
                                                 name='{}_{}'.format(self.name, 3))

        self.max_pool1 = MaxPool1D(pool_size=self.pool_size, strides=1, padding='same',
                                   name='MaxPool1_{}'.format(self.name))
        self.max_pool2 = MaxPool1D(pool_size=self.pool_size, strides=1, padding='same',
                                   name='MaxPool2_{}'.format(self.name))

    def call(self, inputs, **kwargs):
        # 普通卷积
        residual = self.max_pool2(self.bn(self.conv(inputs)))

        # 扩张卷积
        x1 = self.dilation_conv1(inputs)
        x2 = self.dilation_conv2(inputs)
        x3 = self.dilation_conv3(inputs)
        x = tf.keras.layers.add((x1, x2, x3))
        x = self.max_pool1(x)

        x = tf.keras.layers.add((x, residual))
        return x


class MultiWidthDilatedConv(tf.keras.layers.Layer):
    """
    多级横向扩张卷积。设置多级横向扩张卷积
    """

    def __init__(self, multi_dilated_rate, filters=128, padding='same', use_bias=False, activation='relu',
                 pool_size=3, name='multi_width_dilated_conv'):
        """
        :param multi_dilated_rate: (list, mandatory) 扩张卷积层数量。例如: [2,3,4]
        :param filters: (int, optional, default=128) 卷积层过滤器数量
        :param padding: (str, optional, default='same') 卷积层是否进行填充， same 或 valid
        :param use_bias: (bool, optional, default=false) 是否使用 bias
        :param activation: (str, optional, default='relu') 激活函数
        :param pool_size: (int, optional, default=3) 最大池化层 pool_size
        :param name: (str, optional, default='multi_width_dilated_conv')
        :return: (tf.keras.layers.Layer) 序列层
        """
        super(MultiWidthDilatedConv, self).__init__(name=name)

        self.multi_dilated_rate = multi_dilated_rate
        self.filters = filters
        self.padding = padding
        self.use_bias = use_bias
        self.activation = activation
        self.pool_size = pool_size

        self.__init_layer__()

    def __init_layer__(self):
        multi_width_conv = list()
        for dilated_rate in self.multi_dilated_rate:
            width_dilated_conv = WidthDilatedConv(self.filters, self.padding, self.use_bias,
                                                  dilation_rate=dilated_rate,
                                                  pool_size=self.pool_size,
                                                  name='{}_rate_{}'.format(self.name, dilated_rate))
            multi_width_conv.append(width_dilated_conv)
            multi_width_conv.append(Activation(self.activation))

        self.multi_width_dilated_conv = tf.keras.Sequential(multi_width_conv)
        # print(self.multi_width_dilated_conv.layers)
        self.conv = Conv1D(self.filters, kernel_size=1, strides=1, padding=self.padding, use_bias=self.use_bias,
                           name='Conv1D_{}'.format(self.name))
        self.bn = BatchNormalization(name='Bn_{}'.format(self.name))

    def call(self, inputs, **kwargs):
        residual = self.conv(inputs)
        x = self.multi_width_dilated_conv(inputs)

        # 残差叠加
        x = self.bn(tf.keras.layers.add((x, residual)))

        return x


class DepthMultiWidthDilatedConv(tf.keras.layers.Layer):
    """
    深度多级横向扩张卷积。由多个多级横向扩张卷积组成。
    """

    def __init__(self, depth, multi_dilated_rate, filters, padding='same', use_bias=False, activation='relu',
                 pool_size=3, name='depth_multi_width_dilated_conv'):
        """
        初始化
        :param depth: (int, mandatory) 深度。例如 depth=3 表示有 3 个多级横向扩张卷积
        :param multi_dilated_rate: (list, mandatory) 扩张率。例如[2,3,4]
        :param filters: (list, mandatory) 每个深度对应的过滤器大小。例如[128,256,512]。要求：len(filters) == depth
        :param padding: (str, optional, default='same')
        :param use_bias: (bool, optional, default=False)
        :param activation: (str, optional, default='relu')
        :param pool_size: (int, optional, default=3)
        :param name:
        """
        super(DepthMultiWidthDilatedConv, self).__init__(name=name)
        assert isinstance(filters, list), ParameterError('filters must be a list or tuple')
        assert len(filters) == depth, \
            ParameterError("depth:{} and filter length:{} must be equal!".format(depth, len(filters)))
        self.depth = depth
        self.multi_dilated_rate = multi_dilated_rate
        self.filters = filters
        self.padding = padding
        self.use_bias = use_bias
        self.activation = activation
        self.pool_size = pool_size

        self.__init_layer__()

    def __init_layer__(self):
        depth_multi_width_dilated_conv = list()
        for i in range(self.depth):
            multi_width_dilated_conv = MultiWidthDilatedConv(self.multi_dilated_rate,
                                                             filters=self.filters[i],
                                                             padding=self.padding,
                                                             use_bias=self.use_bias,
                                                             activation=self.activation,
                                                             pool_size=self.pool_size,
                                                             name='{}_{}'.format(self.name, i))
            depth_multi_width_dilated_conv.append(multi_width_dilated_conv)

        self.layer = tf.keras.Sequential(depth_multi_width_dilated_conv)

    def call(self, inputs, **kwargs):
        x = self.layer(inputs)
        return x


class WidthConv(tf.keras.layers.Layer):
    """
    横向普通卷积层。区别于横向扩张卷积。主要由 3 个卷积层、1 个池化层、 1 个 add&normal、1 个激活层
    1. 卷积层
        定义 kernel_size=[1,3,5] 的三个卷积层。strides = 1
    2. add & normal
        将三个卷积层的结果进行叠加。然后进行 normal
    3. 最大池化层
        定义一个最大池化层。将上一步的结果进行压缩
    4. 激活层。
        定义激活函数。将上一个的结果输入到激活层
    """

    def __init__(self, filters=128, padding='same', use_bias=False, activation='relu',
                 pool_size=3, pool_strides=2, name='width_conv'):
        super(WidthConv, self).__init__(name=name)
        self.filters = filters
        self.padding = padding
        self.use_bias = use_bias
        self.activation = activation
        self.pool_size = pool_size
        self.pool_strides = pool_strides

        self.__init_layer()

    def __init_layer(self):
        self.conv1 = Conv1D(self.filters, 1, strides=1, padding=self.padding, use_bias=self.use_bias,
                            name='Conv1D_1_{}'.format(self.name))
        self.conv2 = Conv1D(self.filters, 3, strides=1, padding=self.padding, use_bias=self.use_bias,
                            name='Conv1D_2_{}'.format(self.name))

        # 两个 3*3 的卷积层可构成一个 5*5 的卷积层。同时可以减少参数
        # self.conv3_1 = Conv1D(self.filters, 3, strides=1, padding=self.padding, use_bias=self.use_bias,
        #                       name='Conv1D_3_1_{}'.format(self.name))
        # self.conv3_2 = Conv1D(self.filters, 3, strides=1, padding=self.padding, use_bias=self.use_bias,
        #                       name='Conv1D_3_1_{}'.format(self.name))
        self.conv3 = Conv1D(self.filters, 5, strides=1, padding=self.padding, use_bias=self.use_bias,
                            name='Conv1D_3_{}'.format(self.name))

        self.bn = BatchNormalization(name='Bn_{}'.format(self.name))
        self.max_pool = MaxPool1D(pool_size=self.pool_size, strides=self.pool_strides, padding=self.padding,
                                  name='MaxPool_{}'.format(self.name))
        self.activation_func = Activation(self.activation)

    def call(self, inputs, **kwargs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(inputs)
        x3 = self.conv3(inputs)
        # x3 = self.conv3_2(self.conv3_1(inputs))

        x = self.bn(tf.keras.layers.add((x1, x2, x3)))
        x = self.max_pool(x)
        x = self.activation_func(x)

        return x


class DepthWidthConv(tf.keras.layers.Layer):
    """ 深度横向卷积。由多个横向卷积组成 """

    def __init__(self, depth, filters, padding='same', use_bias=False, activation='relu', pool_size=3,
                 pool_strides=2, name='depth_width_conv'):
        """
        初始化
        :param depth: (int, mandatory) 深度。例如 depth=3 表示有 3 个多级横向扩张卷积
        :param filters: (list, mandatory) 每个深度对应的过滤器大小。例如[128,256,512]。要求：len(filters) == depth
        :param padding: (str, optional, default='same')
        :param use_bias: (bool, optional, default=False)
        :param activation: (str, optional, default='relu')
        :param pool_size: (int, optional, default=3)
        :param pool_strides: (int, optional, default=2)
        :param name:
        """
        super(DepthWidthConv, self).__init__(name=name)
        assert isinstance(filters, list), ParameterError('filters must be a list or tuple')
        assert len(filters) == depth, \
            ParameterError("depth:{} and filter length:{} must be equal!".format(depth, len(filters)))
        self.depth = depth
        self.filters = filters
        self.padding = padding
        self.use_bias = use_bias
        self.activation = activation
        self.pool_size = pool_size
        self.pool_strides = pool_strides

        self.__init_layer__()

    def __init_layer__(self):
        depth_layer = list()
        for i in range(self.depth):
            width_conv = WidthConv(self.filters[i], self.padding,
                                   use_bias=self.use_bias,
                                   activation=self.activation,
                                   pool_size=self.pool_size,
                                   pool_strides=self.pool_strides,
                                   name=self.name)
            depth_layer.append(width_conv)

        self.layer = tf.keras.Sequential(depth_layer)

    def call(self, inputs, **kwargs):
        x = self.layer(inputs)
        return x


class ExitTimeDistributed(tf.keras.layers.Layer):
    """
    密集连接层。 由两个 TimeDistributed 组成。
    """

    def __init__(self, hidden_size, output_size, activation='relu', use_bias=True,
                 dropout_rate=0.5, l1=1.0e-5, l2=1.0e-5, name='Exit_TimeDistributed'):
        """
        初始化
        :param hidden_size: (int, mandatory) 隐藏层大小
        :param output_size: (int, mandatory) 输出大小
        :param activation: (str, optional, default='relu') 激活函数
        :param use_bias: (bool, optional, default=True) 是否使用 bias
        :param dropout_rate: (float, optional, default=0.5)
        :param l1: (float, optional, default=1.0e-5)
        :param l2: (float, optional, default=1.0e-5)
        :param name:
        """
        super(ExitTimeDistributed, self).__init__(name=name)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.l1 = l1
        self.l2 = l2

        self.l1_regular = tf.keras.regularizers.l1(self.l1)
        self.l2_regular = tf.keras.regularizers.l2(self.l2)

        self.__init_layer()

    def __init_layer(self):
        dn1 = Dense(self.hidden_size,
                    activation=self.activation,
                    use_bias=self.use_bias,
                    kernel_regularizer=self.l1_regular,
                    bias_regularizer=self.l1_regular,
                    activity_regularizer=self.l2_regular,
                    name='Dense_1_{}'.format(self.name))
        self.time_distributed1 = TimeDistributed(dn1, name='TimeDistributed_1_{}'.format(self.name))

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, noise_shape=(1, 1, self.hidden_size))
        dn2 = Dense(self.output_size,
                    activation=None,
                    use_bias=self.use_bias,
                    kernel_regularizer=self.l1_regular,
                    bias_regularizer=self.l1_regular,
                    activity_regularizer=None,
                    name='Dense_2_{}'.format(self.name))
        self.time_distributed2 = TimeDistributed(dn2, name='TimeDistributed_2_{}'.format(self.name))

    def call(self, inputs, **kwargs):
        training = kwargs['training']

        x = self.time_distributed1(inputs)
        x = self.dropout(x, training=training)
        x = self.time_distributed2(x)

        return x


class TransverseNet(tf.keras.Model):
    """
    Transverse 网络。 由 stem、DepthMultiDilatedConv、DepthWidthConv、TimeDistributed、激活层softmax 组成
    """

    def __init__(self, dilated_conv_depth,
                 width_conv_depth,
                 multi_dilated_rate,
                 dilated_conv_filters,
                 width_conv_filters,
                 dn_hidden_size,
                 dn_output_size,
                 dropout_rate=0.5,
                 l1=1.0e-5,
                 l2=1.0e-5,
                 activation='relu'):
        """
        初始化
        :param dilated_conv_depth: (int, mandatory) 扩张卷积层深度
        :param width_conv_depth: (int, mandatory) 普通卷积层深度
        :param multi_dilated_rate: (list, mandatory) 扩张卷积的扩张率。例如：[2，3，4]
        :param dilated_conv_filters: (list, mandatory) 扩张卷积过滤器数量。
        注意 len(dilated_conv_filters) == dilated_conv_depth
        例如:当dilated_conv_depth=3 时， dilated_conv_filters:[128,256,512]
        :param width_conv_filters: (list, mandatory) 普通卷积过滤器数量。
        注意：len(width_conv_filters) == width_conv_depth
        :param dn_hidden_size: (int, mandatory) TimeDistribute 隐藏层大小
        :param dn_output_size: (int, mandatory) TimeDistribute 输出层大小
        :param dropout_rate: (float, optional, default=0.5) 随机 dropout
        :param l1: (float, optional, default=1.0e-5) L1 正则化系数
        :param l2: (float, optional, default=1.0e-5) L2 正则化系数
        :param activation: (str, optional, default='relu') 激活函数
        """
        super(TransverseNet, self).__init__()
        self.dilated_conv_depth = dilated_conv_depth
        self.width_conv_depth = width_conv_depth
        self.multi_dilated_rate = multi_dilated_rate
        self.dilated_conv_filters = dilated_conv_filters
        self.width_conv_filters = width_conv_filters
        self.dn_hidden_size = dn_hidden_size
        self.dn_output_size = dn_output_size
        self.dropout_rate = dropout_rate
        self.l1 = l1
        self.l2 = l2
        self.activation = activation

        self.__init_layer__()

    def __init_layer__(self):
        self.depth_width_pool_size = 3
        self.depth_width_pool_strides = 2

        self.stem = stem_layer(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False, activation='relu')
        self.depth_multi_width_dilated_conv_layer = DepthMultiWidthDilatedConv(self.dilated_conv_depth,
                                                                               self.multi_dilated_rate,
                                                                               filters=self.dilated_conv_filters,
                                                                               padding='same',
                                                                               use_bias=False,
                                                                               activation=self.activation,
                                                                               pool_size=3,
                                                                               name='depth_multi_width_dilated_conv')
        self.depth_width_conv_layer = DepthWidthConv(self.width_conv_depth,
                                                     self.width_conv_filters,
                                                     padding='same',
                                                     use_bias=False,
                                                     activation=self.activation,
                                                     pool_size=self.depth_width_pool_size,
                                                     pool_strides=self.depth_width_pool_strides,
                                                     name='depth_width_conv')
        # 计算通过 depth_width_conv 层之后的序列大小
        self.width_conv_length = tf.keras.layers.Lambda(lambda x: computer_conv_size(x, self.depth_width_pool_size,
                                                                                     self.depth_width_pool_strides, 0))

        self.time_distributed = ExitTimeDistributed(self.dn_hidden_size,
                                                    self.dn_output_size,
                                                    activation='relu',
                                                    use_bias=True,
                                                    dropout_rate=self.dropout_rate,
                                                    l1=self.l1,
                                                    l2=self.l2,
                                                    name='Exit_TimeDistributed')

        self.activation_func = Activation('softmax', name='softmax')

    def call(self, inputs, sequence_length=None, training=None, mask=None):
        x = self.stem(inputs)
        x = self.depth_multi_width_dilated_conv_layer(x)
        x = self.depth_width_conv_layer(x)
        x = self.time_distributed(x, training=training)
        x = self.activation_func(x)

        # 计算输出卷积层大小
        output_length = sequence_length
        for _ in range(self.width_conv_depth):
            output_length = self.width_conv_length(output_length)

        return x, output_length


class WarmupDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    自定义学习率时间表。通过训练step调节学习率大小。 warmup_steps 定义训练steps == warmup_steps 的时候。学习率开始衰减
    公式： lr = lr * min((steps ** -0.5), steps * (warmup_steps ** -1.5))
    """

    def __init__(self, learning_rate, warmup_steps, name='warmup_decay'):
        super(WarmupDecay, self).__init__()
        self.init_learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.name = name

    def __call__(self, steps, *args, **kwargs):
        return self.init_learning_rate * tf.math.minimum((steps ** -0.5), steps * (self.warmup_steps ** -1.5))

    def get_config(self):
        return {
            'name': self.name,
            'learning_rate': self.init_learning_rate,
            'warmup_steps': self.warmup_steps
        }


class AcousticTransverseNet(object):

    def __init__(self, input_dim,
                 output_vocab_size,
                 dn_hidden_size,
                 dilated_conv_depth,
                 width_conv_depth,
                 multi_dilated_rate,
                 dilated_conv_filters,
                 width_conv_filters,
                 dropout_rate=0.5,
                 l1=1.0e-5,
                 l2=1.0e-5,
                 activation='relu',
                 learning_rate=0.001,
                 warmup_steps=1000,
                 optimizer_beta_1=0.9,
                 optimizer_beta_2=0.999,
                 optimizer_epsilon=1.0e-7,
                 ckpt_dir='./transverse_net',
                 ckpt_max_to_keep=3):
        """
        初始化
        :param input_dim: (int, mandatory) 输入数据的维度
        :param output_vocab_size: (int, mandatory) 输出词汇数量
        :param dn_hidden_size: (int, mandatory) 全连接层大小
        :param dilated_conv_depth: (int, mandatory) 扩张卷积层深度
        :param width_conv_depth: (int, mandatory) 普通卷积层深度
        :param multi_dilated_rate: (list, mandatory) 扩张卷积的扩张率。例如：[1，2，3]。
        :param dilated_conv_filters: (list, mandatory) 扩张卷积过滤器数量。
        注意 len(dilated_conv_filters) == dilated_conv_depth
        如:当dilated_conv_depth=3 时， dilated_conv_filters:[128,256,512]
        :param width_conv_filters: (list, mandatory) 普通卷积过滤器数量。注意：len(width_conv_filters) == width_conv_depth
        :param dropout_rate: (float, optional, default=0.5) 在0到1之间浮动。要降低的输入单位的分数。
        :param l1: (float, optional, default=1.0e-5) L1 正则化系数
        :param l2: (float, optional, default=1.0e-5) L2 正则化系数
        :param activation: (str, optional, default='relu') 层级激活函数
        :param learning_rate: (float, optional, default=0.001) 学习率
        :param warmup_steps: (float, optional, default=1000) 指定学习率下降的step数量。
        例如：当 warmup_steps = 100 时。将在 step=100 时候学习率开始下降
        :param optimizer_beta_1: (float, optional, default=0.9) Adam 优化器 beta_1
        :param optimizer_beta_2: (float, optional, default=0.999) Adam 优化器 beta_2
        :param optimizer_epsilon: (float, optional, default=1.0e-7) Adam 优化器 epsilon
        :param ckpt_dir: (str, optional, default='./transverse_net') 检查点保存目录
        :param ckpt_max_to_keep: (int, optional, default=3) 控制保存检查点的数量
        """
        self.input_dim = input_dim
        self.output_vocab_size = output_vocab_size
        self.dn_hidden_size = dn_hidden_size
        self.dilated_conv_depth = dilated_conv_depth
        self.width_conv_depth = width_conv_depth
        self.multi_dilated_rate = multi_dilated_rate
        self.dilated_conv_filters = dilated_conv_filters
        self.width_conv_filters = width_conv_filters
        self.dropout_rate = dropout_rate
        self.l1 = l1
        self.l2 = l2
        self.activation = activation

        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.optimizer_beta_1 = optimizer_beta_1
        self.optimizer_beta_2 = optimizer_beta_2
        self.optimizer_epsilon = optimizer_epsilon

        self.ckpt_dir = ckpt_dir
        self.ckpt_max_to_keep = ckpt_max_to_keep

        self.__init_optimizer__()
        self.__init_transverse_net__()
        self.__init_ckpt__()
        self.__init_metrics__()

        # 保存每轮次的训练的最小损失
        self.minimum_train_loss = None
        self.minimum_valid_loss = None

    def __init_optimizer__(self):
        """ 初始化优化器 """
        lr = WarmupDecay(learning_rate=self.learning_rate, warmup_steps=self.warmup_steps)
        self.optimizer = tf.keras.optimizers.Adam(lr, beta_1=self.optimizer_beta_1, beta_2=self.optimizer_beta_2,
                                                  epsilon=self.optimizer_epsilon, name='Adam')

    def __init_transverse_net__(self):
        """ 初始化 Transverse 网络"""
        self.transverse = TransverseNet(dilated_conv_depth=self.dilated_conv_depth,
                                        width_conv_depth=self.width_conv_depth,
                                        multi_dilated_rate=self.multi_dilated_rate,
                                        dilated_conv_filters=self.dilated_conv_filters,
                                        width_conv_filters=self.width_conv_filters,
                                        dn_hidden_size=self.dn_hidden_size,
                                        dn_output_size=self.output_vocab_size,
                                        dropout_rate=self.dropout_rate,
                                        l1=self.l1,
                                        l2=self.l2,
                                        activation=self.activation)

        test_data = tf.random.uniform((1, 512, self.input_dim), dtype=tf.float32)
        # test_data = tf.random.uniform((1, 512, self.input_dim), dtype=tf.float64)
        # input_length = tf.random.uniform((1, 1), minval=512, maxval=513, dtype=tf.int32)
        input_length = tf.random.uniform((1, 1), minval=512, maxval=513, dtype=tf.int32)
        self.transverse(test_data, input_length)

    def __init_ckpt__(self):
        """ 初始化检查点 """
        self.checkpoint = tf.train.Checkpoint(transverse=self.transverse, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.ckpt_dir,
                                                             max_to_keep=self.ckpt_max_to_keep)

    def __init_metrics__(self):
        """ 初始化评估指标 """
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')

    def summary(self):
        return self.transverse.summary()

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, input_length, y, label_length):
        """
        训练
        :param x: (tensor, mandatory) 训练数据. shape: [batch_size, seq_len, features]
        :param input_length: (tensor, mandatory). 输入数据集的长度 shape: [batch_size, 1]
        :param y: (tensor, mandatory). 标签数据。 shape: [batch_size, max_length]
        :param label_length: (tensor, mandatory) 标签数据集长度. shape： [batch_size, 1]
        :return:
        """
        with tf.GradientTape() as tape:
            prediction, output_length = self.transverse(x, sequence_length=input_length, training=True)
            ctc_loss = tf.keras.backend.ctc_batch_cost(y_true=y, y_pred=prediction, input_length=output_length,
                                                       label_length=label_length)
            regularizer_loss = tf.add_n(self.transverse.losses)
            loss = ctc_loss + regularizer_loss

        gradients = tape.gradient(loss, self.transverse.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transverse.trainable_variables))

        self.train_loss(loss)

    @tf.function(experimental_relax_shapes=True)
    def valid_step(self, x, input_length, y, label_length):
        """
        验证
        :param x: (tensor, mandatory) 训练数据. shape: [batch_size, seq_len, features]
        :param input_length: (tensor, mandatory). 输入数据集的长度 shape: [batch_size, 1]
        :param y: (tensor, mandatory). 标签数据。 shape: [batch_size, max_length]
        :param label_length: (tensor, mandatory) 标签数据集长度. shape： [batch_size, 1]
        :return:
        """
        prediction, output_length = self.transverse(x, sequence_length=input_length, training=False)
        ctc_loss = tf.keras.backend.ctc_batch_cost(y_true=y, y_pred=prediction, input_length=output_length,
                                                   label_length=label_length)
        self.valid_loss(ctc_loss)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, x, input_length, y, label_length):
        """
        测试
        :param x: (tensor, mandatory) 训练数据. shape: [batch_size, seq_len, features]
        :param input_length: (tensor, mandatory). 输入数据集的长度 shape: [batch_size, 1]
        :param y: (tensor, mandatory). 标签数据。 shape: [batch_size, max_length]
        :param label_length: (tensor, mandatory) 标签数据集长度. shape： [batch_size, 1]
        :return:
        """
        prediction, output_length = self.transverse(x, sequence_length=input_length, training=False)
        ctc_loss = tf.keras.backend.ctc_batch_cost(y_true=y, y_pred=prediction, input_length=output_length,
                                                   label_length=label_length)
        self.test_loss(ctc_loss)

    def save_checkpoint(self):
        """ 保存检查点 """
        self.checkpoint_manager.save()
        print("save checkpoint: {}".format(self.checkpoint_manager.latest_checkpoint))

    def train_save_checkpoint(self):
        """
        训练中保存检查点。当每轮次的 train_loss < minimum_train_loss 和 valid_loss < minimum_valid_loss 时候保存检查点
        """
        is_save = False
        if self.minimum_train_loss is None and self.minimum_valid_loss is None:
            self.minimum_train_loss = self.train_loss.result()
            self.minimum_valid_loss = self.valid_loss.result()
            is_save = True
        else:
            value = tf.math.logical_and(self.minimum_train_loss > self.train_loss.result(),
                                        self.minimum_valid_loss > self.valid_loss.result())
            if value:
                self.minimum_train_loss = self.train_loss.result()
                self.minimum_valid_loss = self.valid_loss.result()
                is_save = True

        if is_save:
            self.save_checkpoint()

    def fit(self, dataset, epochs=10, validation_split=0.1):
        """
        拟合
        :param dataset: (tf.data.Dataset, mandatory) 训练数据集
        :param epochs: (int, optional, default=10) 训练轮次
        :param validation_split: (float, optional, default=0.1) 划分验证数据集比例
        :return: (list) 每轮次的训练损失和验证损失
        """
        train_sample_size, valid_sample_size = get_sample_size(dataset, validation_split)
        train_loss_save, valid_loss_save = list(), list()

        print("start fit ...")
        for e in range(epochs):
            self.train_loss.reset_states()
            self.valid_loss.reset_states()

            # train_dataset = dataset.skip(valid_sample_size).take(train_sample_size)
            train_dataset = dataset.skip(valid_sample_size)
            valid_dataset = dataset.take(valid_sample_size)

            for step, (x, input_length, y, label_length) in enumerate(train_dataset):
                self.train_step(x, input_length, y, label_length)
                output = template(e + 1, epochs, step + 1, train_sample_size, self.train_loss.result(), head='Train')

                sys.stdout.write('\r' + output)
                sys.stdout.flush()
            train_loss_save.append(self.train_loss.result())
            print()

            for step, (x, input_length, y, label_length) in enumerate(valid_dataset):
                self.valid_step(x, input_length, y, label_length)
                output = template(e + 1, epochs, step + 1, valid_sample_size, self.valid_loss.result(), head='Valid')

                sys.stdout.write('\r' + output)
                sys.stdout.flush()
            valid_loss_save.append(self.valid_loss.result())
            print()

            self.train_save_checkpoint()
            print()

        return train_loss_save, valid_loss_save

    def fit_generator(self, gen, batch_size=8, shuffle=True, epochs=10, validation_split=0.1, args=None):
        """
        拟合。使用数据生成器拟合
        :param gen: (Generator, mandatory) 数据生成器
        :param batch_size: (int, optional, default=8) 批量样本
        :param shuffle: (bool, optional, default=True) 是否洗牌
        :param epochs: (int, optional, default=8) 训练轮次
        :param validation_split: (float, optional, default=0.1) 划分验证数据集比例
        :param args: (list, optional, default=None) 需要传入生成器的参数列表
        :return: (list) 每轮次的训练损失和验证损失
        """
        dataset = packaging_data_generator(gen, shuffle=shuffle, batch_size=batch_size, args=args)
        train_loss_save, valid_loss_save = self.fit(dataset, epochs=epochs, validation_split=validation_split)
        return train_loss_save, valid_loss_save

    def eval(self, dataset):
        """
        评估
        :param dataset: (tf.data.Dataset, mandatory) 数据集
        :return: (list) 每个step的损失
        """
        sample_size = len(list(dataset))

        test_loss_save = list()
        print("start eval ...")
        self.test_loss.reset_states()
        for step, (x, input_length, y, label_length) in enumerate(dataset):
            self.test_step(x, input_length, y, label_length)

            output = template(1, 1, step + 1, sample_size, self.test_loss.result(), head='Test')
            sys.stdout.write('\r' + output)
            sys.stdout.flush()

            test_loss_save.append(self.test_loss.result())
        return test_loss_save

    def predict(self, x, input_length, beam_width=3, top_paths=1, greedy=True):
        """
        预测样本。采用集束搜索
        :param x: (array, mandatory) 样本数据。特征和输入长度。shape:(1, sequence_len, feature)
        :param input_length: (array, mandatory) 输入长度。shape:(1,1)
        :param beam_width: (int, optional, default=3) 集束搜索的宽度
        :param top_paths: (int, optional, default=1) 返回的解码序列数量
        :param greedy: (bool, optional, default=True) 如果为 True 则使用贪婪检索，否则使用集束检索
        :return: (array) 解码序列。 返回 array 类型的序列和 tensor 类型的值
        """
        if greedy:
            sequence_array, value = self.predict_sample_greedy(x, input_length)
        else:
            sequence_array, value = self.predict_sample_beam(x, input_length, beam_width, top_paths)

        return sequence_array, value

    def predict_sample_greedy(self, x, input_length):
        """
        预测样本。采用贪婪搜索
        :param x: (array, mandatory) 样本数据。特征和输入长度 。shape:(1, sequence_len, feature)
        :param input_length: (array, mandatory) 输入长度。shape:(1,1)
        :return: (array) 解码序列。返回 array 类型的序列和 tensor 类型的值
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        input_length = tf.convert_to_tensor(input_length, dtype=tf.int32)

        prediction, output_length = self.transverse(x, input_length)
        output_length = tf.reshape(output_length, shape=[1])
        sequence_index, value = tf.keras.backend.ctc_decode(y_pred=prediction, input_length=output_length, greedy=True)
        assert len(sequence_index) == 1, \
            UnknownError("{} return sequence not equal to 1.".format(self.predict_sample_greedy.__name__))

        sequence_index = sequence_index[0].numpy().flatten()

        return sequence_index, value

    def predict_sample_beam(self, x, input_length, beam_width=3, top_paths=1):
        """
        预测样本。采用集束搜索
        :param x: (array, mandatory) 样本数据。特征和输入长度。shape:(1, sequence_len, feature)
        :param input_length: (array, mandatory) 输入长度。shape:(1,1)
        :param beam_width: (int, optional, default=3) 集束搜索的宽度
        :param top_paths: (int, optional, default=1) 返回的解码序列数量
        :return: (array) 解码序列。 返回 array 类型的序列和 tensor 类型的值
        """

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        input_length = tf.convert_to_tensor(input_length, dtype=tf.int32)

        prediction, output_length = self.transverse(x, input_length)
        output_length = tf.reshape(output_length, shape=[1])
        sequence_index, value = tf.keras.backend.ctc_decode(y_pred=prediction, input_length=output_length,
                                                            greedy=False, beam_width=beam_width, top_paths=top_paths)

        sequence_array = [sequence_index[i].numpy().flatten() for i in range(top_paths)]

        return sequence_array, value

    def dump_config(self, file='acoustic_transverse.json'):
        """
        备份配置信息
        :param file: (str, optional, default='acoustic_transverse.json') 配置信息文件名
        :return:
        """
        config = {
            'class': AcousticTransverseNet.__name__,
            'input_dim': self.input_dim,
            'output_vocab_size': self.output_vocab_size,
            'dn_hidden_size': self.dn_hidden_size,
            'dilated_conv_depth': self.dilated_conv_depth,
            'width_conv_depth': self.width_conv_depth,
            'multi_dilated_rate': self.multi_dilated_rate,
            'dilated_conv_filters': self.dilated_conv_filters,
            'width_conv_filters': self.width_conv_filters,
            'dropout_rate': self.dropout_rate,
            'l1': self.l1,
            'l2': self.l2,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'optimizer_beta_1': self.optimizer_beta_1,
            'optimizer_beta_2': self.optimizer_beta_2,
            'optimizer_epsilon': self.optimizer_epsilon,
            'ckpt_dir': self.ckpt_dir,
            'ckpt_max_to_keep': self.ckpt_max_to_keep
        }

        Writer.check_path(file)
        with open(file, 'w') as f:
            json.dump(config, f, indent=4)

        print("writer config over! file: {}".format(file))

    @staticmethod
    def from_config(config):
        """
        根据配置还原模型
        :param config: (str, mandatory) 配置文件
        :return: (AcousticTransverseNet) 模型对象
        """
        assert os.path.isfile(config), FileNotFoundException(config)

        with open(config, 'r') as f:
            data = json.load(f)
        assert data['class'] == AcousticTransverseNet.__name__, ParameterError(
            'config:{} class:{} error, expect get{}'.format(config, data['class'], AcousticTransverseNet.__name__))

        acoustic = AcousticTransverseNet(
            input_dim=data['input_dim'],
            output_vocab_size=data['output_vocab_size'],
            dn_hidden_size=data['dn_hidden_size'],
            dilated_conv_depth=data['dilated_conv_depth'],
            width_conv_depth=data['width_conv_depth'],
            multi_dilated_rate=data['multi_dilated_rate'],
            dilated_conv_filters=data['dilated_conv_filters'],
            width_conv_filters=data['width_conv_filters'],
            dropout_rate=data['dropout_rate'],
            l1=data['l1'],
            l2=data['l2'],
            activation=data['activation'],
            learning_rate=data['learning_rate'],
            warmup_steps=data['warmup_steps'],
            optimizer_beta_1=data['optimizer_beta_1'],
            optimizer_beta_2=data['optimizer_beta_2'],
            optimizer_epsilon=data['optimizer_epsilon'],
            ckpt_dir=data['ckpt_dir'],
            ckpt_max_to_keep=data['ckpt_max_to_keep']
        )

        return acoustic

    @staticmethod
    def restore(config, checkpoint):
        """
        还原模型
        :param config: (str, mandatory) 配置文件
        :param checkpoint: (str, mandatory) 检查点目录
        :return: (AcousticTransverseNet) 模型对象
        """
        acoustic = AcousticTransverseNet.from_config(config)

        if tf.train.latest_checkpoint(checkpoint):
            acoustic.checkpoint.restore(tf.train.latest_checkpoint(checkpoint)).expect_partial()
            output = 'restore checkpoint from: {}'.format(tf.train.latest_checkpoint(checkpoint))
        else:
            output = "Not found checkpoint Initializing from scratch"

        print(Color.green(output))
        return acoustic


def virtual_data_generator(size, x_length, y_length, x_dim, y_range=1000):
    for i in range(size):
        input_length = np.random.randint(low=x_length - 100, high=x_length, size=[1], dtype=np.int32)

        x = np.zeros((x_length, x_dim), dtype=np.float32)
        x_data = np.random.uniform(size=(input_length[0], x_dim)).astype(np.float32)
        x[:x_data.shape[0], :] = x_data

        label_length = np.random.randint(low=y_length - 4, high=y_length, size=[1], dtype=np.int32)
        y = np.ones(y_length) * y_range
        y_data = np.random.randint(low=0, high=y_range, size=[label_length[0]])
        y[:len(y_data)] = y_data

        yield x, input_length.tolist(), y, label_length.tolist()


def virtual_dataset(size, x_length, y_length, x_dim, y_range, batch_size=4, shuffle=True):
    output_shapes = (tf.TensorShape([None, x_dim]),
                     tf.TensorShape([None]),
                     tf.TensorShape([None]),
                     tf.TensorShape([None]))
    dataset = tf.data.Dataset.from_generator(virtual_data_generator,
                                             args=[size, x_length, y_length, x_dim, y_range],
                                             output_types=(tf.float32, tf.int32, tf.int32, tf.int32),
                                             output_shapes=output_shapes)
    if shuffle:
        dataset = dataset.shuffle(batch_size * 2).batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)

    return dataset


def test_module_func(operation):
    if operation == stem_layer.__name__:
        stem = stem_layer()
        data = tf.random.uniform((32, 128, 256))
        output = stem(data)
        print(output.shape)
        assert output.shape == (32, 128, 64)

    elif operation == WidthDilatedConv.__name__:
        width_dilated_conv = WidthDilatedConv()
        data = tf.random.uniform((32, 1208, 256))
        output = width_dilated_conv(data)
        print(output.shape)

    elif operation == MultiWidthDilatedConv.__name__:
        layer = MultiWidthDilatedConv([2, 3, 4])
        data = tf.random.uniform((32, 1208, 256))
        output = layer(data)
        print(output.shape)

    elif operation == DepthMultiWidthDilatedConv.__name__:
        layer = DepthMultiWidthDilatedConv(3, multi_dilated_rate=[2, 3, 4], filters=[128, 256, 512])
        data = tf.random.uniform((32, 1208, 256))
        output = layer(data)
        print(output.shape)

    elif operation == WidthConv.__name__:
        layer = WidthConv(filters=256)
        data = tf.random.uniform((32, 1208, 128))
        output = layer(data)
        print(output.shape)

    elif operation == DepthWidthConv.__name__:
        layer = DepthWidthConv(3, [128, 256, 512])
        data = tf.random.uniform((32, 1208, 128))
        output = layer(data)
        print(output.shape)

    elif operation == ExitTimeDistributed.__name__:
        layer = ExitTimeDistributed(256, 512)
        data = tf.random.uniform((32, 512, 128))
        output = layer(data, training=True)
        print(output.shape)

    elif operation == TransverseNet.__name__:
        model = TransverseNet(dilated_conv_depth=2,
                              width_conv_depth=3,
                              multi_dilated_rate=[2, 3, 4],
                              dilated_conv_filters=[128, 256],
                              width_conv_filters=[512, 512, 512],
                              dn_hidden_size=1024,
                              dn_output_size=1000,
                              dropout_rate=0.5,
                              l1=1.0e-5,
                              l2=1.0e-5,
                              activation='relu')

        data = tf.random.uniform((4, 1025, 13))
        input_length = tf.random.uniform((4, 1), minval=1024, maxval=1025, dtype=tf.int32)
        output, _ = model(data, input_length)

        print(output.shape)
        print(model.summary())
        print("regularizer losses: ", model.losses)

    elif operation == WarmupDecay.__name__:
        lr_warmup = WarmupDecay(learning_rate=0.1, warmup_steps=100)
        plt.figure(figsize=(8, 6))
        plt.plot(lr_warmup(tf.range(300, dtype=tf.float32)))
        plt.show()

    elif operation == virtual_data_generator.__name__:
        size = 100
        x_length = 1024
        y_length = 32
        x_dim = 13
        y_range = 1000
        for x, x_len, y, y_len in virtual_data_generator(size, x_length, y_length, x_dim, y_range):
            print("x shape:", x.shape)
            print("x length: ", x_len)
            print("y shape: ", y.shape)
            print("y length: ", y_len)
        # virtual_dataset(size, x_length, y_length, x_dim, y_range, batch_size=4, shuffle=True)

    elif operation == virtual_dataset.__name__:
        size = 8
        x_length = 1024
        y_length = 32
        x_dim = 13
        y_range = 1000
        batch_size = 4
        shuffle = True

        dataset = virtual_dataset(size, x_length, y_length, x_dim, y_range, batch_size=batch_size, shuffle=shuffle)
        for x, x_len, y, y_len in dataset:
            print("x shape: ", x.shape)
            print("x length shape: ", x_len.shape)
            print("y shape: ", y.shape)
            print("y length shape: ", y_len.shape)
            print()

    elif operation == AcousticTransverseNet.fit.__name__:
        size = 100
        x_length = 1024
        y_length = 32
        x_dim = 13
        y_range = 256
        batch_size = 4
        shuffle = True
        dataset = virtual_dataset(size, x_length, y_length, x_dim, y_range, batch_size, shuffle)
        model = AcousticTransverseNet(input_dim=x_dim,
                                      output_vocab_size=y_range + 1,
                                      dn_hidden_size=512,
                                      dilated_conv_depth=3,
                                      width_conv_depth=3,
                                      multi_dilated_rate=[2, 3, 4],
                                      dilated_conv_filters=[128, 256, 512],
                                      width_conv_filters=[512, 512, 512],
                                      learning_rate=0.1,
                                      warmup_steps=100)
        print(model.summary())
        model.fit(dataset, epochs=2)

    elif operation == AcousticTransverseNet.eval.__name__:
        size = 100
        x_length = 1024
        y_length = 32
        x_dim = 13
        y_range = 256
        batch_size = 1
        shuffle = False
        dataset = virtual_dataset(size, x_length, y_length, x_dim, y_range, batch_size, shuffle)
        model = AcousticTransverseNet(input_dim=x_dim,
                                      output_vocab_size=y_range + 1,
                                      dn_hidden_size=512,
                                      dilated_conv_depth=3,
                                      width_conv_depth=3,
                                      multi_dilated_rate=[2, 3, 4],
                                      dilated_conv_filters=[128, 256, 512],
                                      width_conv_filters=[512, 512, 512],
                                      learning_rate=0.1,
                                      warmup_steps=100)
        print(model.summary())
        model.eval(dataset)

    elif operation == AcousticTransverseNet.predict.__name__:
        x_dim = 13
        y_range = 256
        model = AcousticTransverseNet(input_dim=x_dim,
                                      output_vocab_size=y_range + 1,
                                      dn_hidden_size=512,
                                      dilated_conv_depth=3,
                                      width_conv_depth=3,
                                      multi_dilated_rate=[2, 3, 4],
                                      dilated_conv_filters=[128, 256, 512],
                                      width_conv_filters=[512, 512, 512],
                                      learning_rate=0.1,
                                      warmup_steps=100)
        x = tf.random.uniform((1, 1025, 13))
        input_length = tf.random.uniform((1, 1), minval=1024, maxval=1025, dtype=tf.int32)
        sequence_array, value = model.predict(x, input_length, greedy=True)
        print("greedy: ", sequence_array)

        print()
        sequence_array, value = model.predict(x, input_length, beam_width=3, top_paths=2, greedy=False)
        print("beam: ", sequence_array)

    elif operation == AcousticTransverseNet.restore.__name__:
        size = 100
        x_length = 1024
        y_length = 32
        x_dim = 13
        y_range = 256
        batch_size = 1
        shuffle = False
        config_file = './transverse/acoustic_transverse.json'
        checkpoint = './transverse/ckpt'
        dataset = virtual_dataset(size, x_length, y_length, x_dim, y_range, batch_size, shuffle)
        model = AcousticTransverseNet(input_dim=x_dim,
                                      output_vocab_size=y_range + 1,
                                      dn_hidden_size=512,
                                      dilated_conv_depth=3,
                                      width_conv_depth=3,
                                      multi_dilated_rate=[2, 3, 4],
                                      dilated_conv_filters=[128, 256, 512],
                                      width_conv_filters=[512, 512, 512],
                                      learning_rate=0.1,
                                      warmup_steps=100,
                                      ckpt_dir=checkpoint)

        model.fit(dataset, epochs=2)
        model.dump_config(config_file)

        print("start restore model ...")
        new_model = AcousticTransverseNet.restore(config_file, checkpoint)
        new_model.fit(dataset, epochs=2)

    elif operation == packaging_data_generator.__name__:
        size = 8
        x_length = 1024
        y_length = 32
        x_dim = 13
        y_range = 256
        batch_size = 1
        shuffle = False
        dataset = packaging_data_generator(virtual_data_generator, args=[size, x_length, y_length, x_dim, y_range],
                                           batch_size=batch_size, shuffle=shuffle)

        for x, input_length, y, label_length in dataset:
            print("x shape: ", x.shape)
            print("input length: ", input_length)
            print("y: ", y)
            print("label length: ", label_length)

    elif operation == AcousticTransverseNet.fit_generator.__name__:
        size = 8
        x_length = 1024
        y_length = 32
        x_dim = 13
        y_range = 256
        batch_size = 1
        shuffle = False
        checkpoint = './transverse/ckpt'
        model = AcousticTransverseNet(input_dim=x_dim,
                                      output_vocab_size=y_range + 1,
                                      dn_hidden_size=512,
                                      dilated_conv_depth=3,
                                      width_conv_depth=3,
                                      multi_dilated_rate=[2, 3, 4],
                                      dilated_conv_filters=[128, 256, 512],
                                      width_conv_filters=[512, 512, 512],
                                      learning_rate=0.1,
                                      warmup_steps=100,
                                      ckpt_dir=checkpoint)
        print(model.summary())
        model.fit_generator(virtual_data_generator, args=[size, x_length, y_length, x_dim, y_range],
                            batch_size=batch_size, shuffle=shuffle, epochs=2)


if __name__ == '__main__':
    # test_module_func(stem_layer.__name__)
    # test_module_func(WidthDilatedConv.__name__)
    # test_module_func(MultiWidthDilatedConv.__name__)
    # test_module_func(DepthMultiWidthDilatedConv.__name__)
    # test_module_func(WidthConv.__name__)
    # test_module_func(DepthWidthConv.__name__)
    # test_module_func(ExitTimeDistributed.__name__)
    # test_module_func(TransverseNet.__name__)
    # test_module_func(WarmupDecay.__name__)
    # test_module_func(virtual_data_generator.__name__)
    # test_module_func(virtual_dataset.__name__)
    test_module_func(AcousticTransverseNet.fit.__name__)
    # test_module_func(AcousticTransverseNet.eval.__name__)
    # test_module_func(AcousticTransverseNet.predict.__name__)
    # test_module_func(AcousticTransverseNet.restore.__name__)
    # test_module_func(AcousticTransverseNet.fit_generator.__name__)
    # test_module_func(packaging_data_generator.__name__)
