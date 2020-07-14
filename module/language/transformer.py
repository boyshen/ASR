# -*- encoding: utf-8 -*-
"""
@file: transformer.py
@time: 2020/6/24 上午9:39
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: 
"""
import sys
import os
import json
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from module.core.utils import template
from module.core.utils import Writer
from module.core.color import Color
from module.core.exception import ParameterError
from module.core.exception import FileNotFoundException


def get_angles(pos, i, d_model):
    """
    获取角度。 根据 PE(d_model, 2i) = sin(pos / 10000^(2i/d_model)) 函数。计算 pos ／ 10000^(2i/d_model)
    :param pos: (array, mandatory)。位置编码。如果位置数量是 10，即有 10 个单位长度的序列。则 pos=[[0],[1]...[9]].shape:[10,1]
    :param i: (array, mandatory)。维度编码。如果维度数量是 512， 即有 512 个维度，则 i = [[0,1,2, ... 511]]. shape:[1,512]
    :param d_model: (int, mandatory) 维度数量。
    :return: (array) 返回 pos / 10000^(2i/d_model) 的计算结果。shape:[len(pos), d_model]
    """
    min_state = 1 / 10000
    # i // 2 即区分奇数和偶数
    angle_rates = min_state ** ((2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(pos, d_model):
    """
    位置编码。 根据transform的位置编码:
        PE(d_model, 2i) = sin(pos / 10000^(2i/d_model))
        PE(d_model, 2i+1) = cos(pos / 10000^(21/d_model))
    参考：https://tensorflow.google.cn/tutorials/text/transformer?hl=zh_cn
    :param pos: (int, mandatory) 需要嵌入的位置数量
    :param d_model: (int, mandatory) 维度数量
    :return: (tensor) shape: [1, pos, d_model]
    """
    angle_rate = get_angles(np.arange(pos)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # 将 sin 用于计算偶数 2i
    angle_rate[:, 0::2] = np.sin(angle_rate[:, 0::2])

    # 将 cos 用于计算奇数 2i+1
    angle_rate[:, 1::2] = np.cos(angle_rate[:, 1::2])

    angle_rate = tf.cast(angle_rate, dtype=tf.float32)
    return angle_rate[tf.newaxis, :, :]


def show_positional_encoding(pos_encoding):
    """
    绘制位置编码图
    :param pos_encoding: (tensor, mandatory) 位置编码。shape：[pos, d_model]
    :return:
    """
    plt.pcolormesh(pos_encoding, cmap='RdBu')
    plt.xlabel('depth')
    plt.ylabel('positional')
    plt.xlim((0, pos_encoding.shape[1]))
    plt.colorbar()
    plt.show()


def create_padding_mask(seq, mask_index=0):
    """
    创建填充掩码。用于遮挡序列中的填充标记。确保该填充标记不会作为输入。
    例如输入序列 [[1,2,3,4,0,0,1,0]] 返回 [[0,0,0,0,1,1,0,1]] 。(默认在 mask=0 的情况下)
    :param seq: (tensor, mandatory) 序列
    :param mask_index: (int, optional, default=0) 掩码值，默认为 0 。即标记序列中属于填充标记的掩码值
    :return: (tensor) 序列。shape = [seq.shape[0], 1, 1, seq.shape[1]]
    """
    seq = tf.cast(tf.math.equal(seq, mask_index), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    前瞻遮挡。用于遮挡一个序列中后续的标记。
    例如：预测第三个词，将仅使用第一个和第二个词。与此类似，预测第四个词，仅使用第一个，第二个和第三个词，
    输入：3，
    输出：
    [[0,1,1],
    [[0,0,1],
    [0,0,0]]
    :param size: (int, mandatory) 序列大小
    :return: (tensor) 掩码. shape: [size, size]。
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_mask(inputs, targets_inputs, mask_index=0):
    """
    创建填充掩码和目标掩码
    :param inputs: (tensor, mandatory) 输入样本数据。shape:[batch_size, seq_len]
    :param targets_inputs: (tensor, mandatory) 目标样本数据，也是标签数据。shape:[batch_size, :-1]
    :param mask_index: (int, optional, default=0) 掩码索引值。
    :return: (tensor) 编码器填充掩码、
                      解码器混合掩码(用于第一个注意力)，
                      解码器填充掩码，用于第二个注意力对编码器输出进行遮挡
    """
    # 编码器填充掩码
    encoder_padding_mask = create_padding_mask(inputs, mask_index)

    # 解码器第一个注意力填充掩码
    decoder_padding_mask = create_padding_mask(targets_inputs, mask_index)
    # 解码器第一个注意力前瞻掩码
    decoder_look_ahead_mask = create_look_ahead_mask(tf.shape(targets_inputs)[1])
    # 将解码器的填充掩码和前瞻掩码进行合并
    decoder_mask = tf.math.maximum(decoder_padding_mask, decoder_look_ahead_mask)

    # 解码器第二个注意力填充掩码。用于对编码器的输出进行遮挡
    decoder_padding_mask = create_padding_mask(inputs, mask_index)

    return encoder_padding_mask, decoder_mask, decoder_padding_mask


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    缩放点乘注意力。也叫自注意力。主要计算注意力权重。
    计算：Attention(q,k,v) = dot(softmax(dot(q,k^T) / sqrt(dk), v) # dot 为矩阵乘. k^T 为 k 的转置
    dk: 例如，假设 Q 和 K 的均值为0，方差为1。它们的矩阵乘积将有均值为0，方差为 dk。
    因此，dk 的平方根被用于缩放（而非其他数值），因为，Q 和 K 的矩阵乘积的均值本应该为 0，方差本应该为1，这样会获得一个更平缓的 softmax。
    mask: 遮挡（mask）与 -1e9（接近于负无穷）相乘。
    这样做是因为遮挡与缩放的 Q 和 K 的矩阵乘积相加，并在 softmax 之前立即应用。
    目标是将这些单元归零，因为 softmax 的较大负数输入在输出中接近于零。
    参考：https://tensorflow.google.cn/tutorials/text/transformer?hl=zh_cn
    :param q: (tensor, mandatory) 查询 Q： shape: (..., seq_len_q, depth)
    :param k: (tensor, mandatory) 键  K： shape: (..., seq_len_k, depth) K的depth == Q的depth
    :param v: (tensor, mandatory) 数值 V: shape:  (..., seq_len_v, depth_v) K的seq_len_k == V的seq_len_v
    :param mask: (tensor, optional, default=None) 掩码
    :return: (tensor) 点积值、softmax的输出即注意力权重
    """
    matmul_output = tf.matmul(q, k, transpose_b=True)

    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_matmul_output = matmul_output / tf.math.sqrt(dk)

    if mask is not None:
        scaled_matmul_output += (mask * -1.0e9)

    attention_weight = tf.nn.softmax(scaled_matmul_output)

    output = tf.matmul(attention_weight, v)
    return output, attention_weight


def point_wise_feed_forward_network(input_dim, output_dim):
    """
    点式前馈网络。即有两个全链接层组成的网络。使用 relu 激活函数
    :param input_dim: (int, mandatory) 输入隐藏层数量
    :param output_dim: (int, mandatory) 输出层数量
    :return: (Sequential) 序列网络
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(input_dim, activation='relu'),
        tf.keras.layers.Dense(output_dim)
    ])


def packaging_dataset(x, y, batch_size=8, shuffle=True):
    """
    包装数据集。将 x 和 y 包装成数据流水线
    :param x: (list, mandatory) 训练数据集
    :param y: (list, mandatory) 标签数据集
    :param batch_size: (int, optional, default=8) 样本大小
    :param shuffle: (bool, optional, default=True) 是否进行洗牌
    :return: (tf.data.Dataset) 数据集
    """
    assert len(x) == len(y), ParameterError("x:{} and y:{} have different data size! ".format(len(x), len(y)))
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(batch_size * 2).batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)

    return dataset


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


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    多头注意力。多头注意力主要由四个部分组成。
    1. Q、K、V 线性层, 并拆分成多头。 多头注意力由 q，k，v 三个输入，q、k、v 分别输入到对应的线性层Q、K、V，并对结果进行拆分成多个头
    2. 缩放点乘注意力。 由函数 scaled_dot_product_attention 完成。
    3. 多头级连。 将拆分的多头进行合并。
    4. 线性输出层。 将上一步的结果输入的线性层。返回输出结果
    """

    def __init__(self, d_model, head_num):
        """
        初始化。 注意：head_num 需要能被 d_model 整除。即 d_model % head_num == 0
        :param d_model: (int, mandatory) 维度数量。通常是 512
        :param head_num: (int, mandatory) 头数量。通常是 8
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % head_num == 0, ParameterError(
            "head_num:{} needs to be able to be d_model:{} division".format(head_num, d_model))
        self.d_model = d_model
        self.head_num = head_num

        self.depth = self.d_model // self.head_num

        self.wq = tf.keras.layers.Dense(self.d_model)
        self.wk = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)

        self.dn = tf.keras.layers.Dense(self.d_model)

    def split_heads(self, inputs, batch_size):
        """
        拆分成多个头。 即如果序列 shape是[1, 50, 512] 则返回: [1, self.head_num, 50, self.d_model//self.head_num]
        :param inputs:(tensor, mandatory) 序列
        :param batch_size: (int, mandatory) 样本大小
        :return:
        """
        # 将序列 shape 转换成 [batch_size, seq_len, head_num, depth]
        output = tf.reshape(inputs, shape=[batch_size, -1, self.head_num, self.depth])

        # 将序列 shape 转换成 [batch_size, head_num, seq_len, depth]
        return tf.transpose(output, perm=[0, 2, 1, 3])

    def call(self, inputs, **kwargs):
        # 获取输入q、k、v、mask
        q, k, v = inputs
        mask = kwargs['mask']
        batch_size = tf.shape(q)[0]

        # 将 q、k、v 序列值输入到线性层 wq、wk、wv 。
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # 拆分成多个头
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # 缩放点乘注意力
        scaled_attention, scaled_weight = scaled_dot_product_attention(q, k, v, mask)

        # 转换序列维度。将shape：[batch, head_num, seq_len, depth] 转换成 [batch, seq_len, head_num, depth]
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # 多头级连
        concat_attention = tf.reshape(scaled_attention, shape=[batch_size, -1, self.d_model])

        # 线性输出层. shape: [batch_size, seq_len, d_model]
        output = self.dn(concat_attention)

        return output, scaled_weight


class EncoderLayer(tf.keras.layers.Layer):
    """
    transformer 编码器层。主要包含两个子层：
    1. 多头注意力 (需要有填充遮挡)
    2. 点式前馈网络
    每个子层都包含一个残差连接和层归一化， 残差网络有助于避免深度网络中梯度消失
    """

    def __init__(self, d_model, heads_num, forward_hidden, dropout_rate=0.1):
        """
        初始化
        :param d_model: (int, mandatory) 维度数量。通常是 512
        :param heads_num: (int, mandatory) 多头数量。通常是 8
        :param forward_hidden: (int, mandatory) 前馈层隐藏数量。通常是1024 或 2048
        :param dropout_rate: (float, optional, default=0.1) 随机 Dropout 的概率
        """
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.heads_num = heads_num
        self.forward_hidden = forward_hidden
        self.dropout_rate = dropout_rate

        self.multi_head_attention = MultiHeadAttention(d_model=self.d_model, head_num=self.heads_num)
        self.feed_forward = point_wise_feed_forward_network(input_dim=self.forward_hidden, output_dim=self.d_model)

        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)

        self.layer_normal1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_normal2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, **kwargs):
        q = inputs
        k = inputs
        v = inputs
        mask = kwargs['mask']
        training = kwargs['training']

        # 多头注意力。 shape: [batch_size, seq_len, d_model]
        attention_output, _ = self.multi_head_attention((q, k, v), mask=mask)
        attention_output = self.dropout1(attention_output, training=training)
        # 残差连接并且归一化。 shape: [batch_size, seq_len, d_model]
        output1 = self.layer_normal1(inputs + attention_output)

        # 点式前馈网络
        forward_output = self.feed_forward(output1)
        forward_output = self.dropout2(forward_output, training=training)
        output2 = self.layer_normal2(output1 + forward_output)
        return output2


class DecoderLayer(tf.keras.layers.Layer):
    """
    transformer 解码层。主要包括三个子层：
    1. 前瞻掩码的多头注意力
    2. 填充掩码的多头注意力。K、V 为编码器的输出。Q 为上一层的输出。
    3. 点式前馈网络
    每一层都包含一个残差连接和归一化层
    """

    def __init__(self, d_model, heads_num, forward_hidden, dropout_rate=0.1):
        """
        初始化
        :param d_model: (int, mandatory) 维度数量。通常是 512
        :param heads_num: (int, mandatory) 多头数量。通常是 8
        :param forward_hidden: (int, mandatory) 前馈层隐藏数量。通常是1024 或 2048
        :param dropout_rate: (float, optional, default=0.1) 随机 Dropout 的概率
        """
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.heads_num = heads_num
        self.forward_hidden = forward_hidden
        self.dropout_rate = dropout_rate

        self.multi_head_attention1 = MultiHeadAttention(d_model=self.d_model, head_num=self.heads_num)
        self.multi_head_attention2 = MultiHeadAttention(d_model=self.d_model, head_num=self.heads_num)

        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(self.dropout_rate)

        self.layer_normal1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_normal2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_normal3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.feed_forward = point_wise_feed_forward_network(self.forward_hidden, self.d_model)

    def call(self, inputs, **kwargs):
        q = inputs
        k = inputs
        v = inputs
        encoder_output = kwargs['encoder_output']

        look_ahead_mask = kwargs['look_ahead_mask']
        padding_mask = kwargs['padding_mask']

        training = kwargs['training']

        attention_output1, attention_weight1 = self.multi_head_attention1((q, k, v), mask=look_ahead_mask)
        attention_output1 = self.dropout1(attention_output1, training=training)
        output1 = self.layer_normal1(inputs + attention_output1)

        attention_output2, attention_weight2 = self.multi_head_attention2((output1, encoder_output, encoder_output),
                                                                          mask=padding_mask)
        attention_output2 = self.dropout2(attention_output2, training=training)
        output2 = self.layer_normal2(output1 + attention_output2)

        forward_output = self.feed_forward(output2)
        forward_output = self.dropout3(forward_output, training=training)
        output3 = self.layer_normal3(output2 + forward_output)

        return output3, attention_weight1, attention_weight2


class Encoder(tf.keras.layers.Layer):
    """
    编码器层。主要包括：
    1. 词嵌入。
    2. 位置编码。
    3. N个编码器层。
    经过词嵌入之后，将词嵌入与位置编码相加。然后输入到编码器层。编码器的输出是解码器的输入。
    """

    def __init__(self, vocab_size, d_model, heads_num, forward_hidden, num_layers, max_positional_encoding,
                 dropout_rate=0.1):
        """
        初始化
        :param vocab_size: (int, mandatory) 词嵌入的输入维度
        :param d_model: (int, mandatory) 维度数量。通常是 512
        :param heads_num: (int, mandatory) 多头注意力的数量
        :param forward_hidden: (int, mandatory) 前馈隐藏层数量
        :param num_layers: (int, mandatory) N 个编码器层
        :param max_positional_encoding: (int, mandatory) 最大位置编码
        :param dropout_rate: (float, optional, default=0.1)
        """
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.heads_num = heads_num
        self.forward_hidden = forward_hidden
        self.num_layers = num_layers
        self.max_positional_encoding = max_positional_encoding
        self.dropout_rate = dropout_rate

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = positional_encoding(self.max_positional_encoding, self.d_model)

        self.n_encoder_layer = []
        for _ in range(self.num_layers):
            self.n_encoder_layer.append(EncoderLayer(self.d_model, self.heads_num, self.forward_hidden,
                                                     self.dropout_rate))

    def call(self, inputs, **kwargs):
        seq_len = tf.shape(inputs)[1]
        mask = kwargs['mask']
        training = kwargs['training']

        # 词嵌入
        embedding_output = self.embedding(inputs)
        embedding_output = embedding_output * tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))
        # 与位置编码相加
        positional_embedding_output = embedding_output + self.positional_encoding[:, :seq_len, :]

        # N 个编码器层
        x = positional_embedding_output
        for i in range(self.num_layers):
            x = self.n_encoder_layer[i](x, training=training, mask=mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """
    解码器层。主要包括：
    1. 词嵌入
    2. 位置编码
    3. N个解码器
    与编码器一样。需要将词嵌入与位置编码相加。解码器的输出是最后线性层的输入。
    """

    def __init__(self, vocab_size, d_model, heads_num, forward_hidden, num_layers, max_positional_encoding,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.heads_num = heads_num
        self.forward_hidden = forward_hidden
        self.num_layers = num_layers
        self.max_positional_encoding = max_positional_encoding
        self.dropout_rate = dropout_rate

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = positional_encoding(self.max_positional_encoding, self.d_model)

        self.n_decoder_layer = []
        for _ in range(self.num_layers):
            self.n_decoder_layer.append(DecoderLayer(self.d_model, self.heads_num, self.forward_hidden,
                                                     self.dropout_rate))

    def call(self, inputs, **kwargs):
        seq_len = tf.shape(inputs)[1]
        encoder_output = kwargs['encoder_output']
        look_ahead_mask = kwargs['look_ahead_mask']
        padding_mask = kwargs['padding_mask']
        training = kwargs['training']

        # 词嵌入
        embedding_output = self.embedding(inputs)
        embedding_output = embedding_output * tf.math.sqrt(tf.cast(self.d_model, dtype=tf.float32))

        # 位置编码
        positional_encoding_output = embedding_output + self.positional_encoding[:, :seq_len, :]

        # N 个解码器层
        attention_weights = []
        x = positional_encoding_output
        for i in range(self.num_layers):
            x, attention_weight1, attention_weight2 = self.n_decoder_layer[i](x, encoder_output=encoder_output,
                                                                              look_ahead_mask=look_ahead_mask,
                                                                              padding_mask=padding_mask,
                                                                              training=training)

            attention_weights.append({'decoder_layer{}_attention_weight1'.format(i): attention_weight1,
                                      'decoder_layer{}_attention_weight2'.format(i): attention_weight2}
                                     )

        return x, attention_weights


class Transformer(tf.keras.Model):
    """
    transformer 模型。包括 编码器、解码器、最后的线性层。
    """

    def __init__(self, input_vocab_size, output_vocab_size, d_model, heads_num, forward_hidden, num_layers,
                 input_max_positional, target_max_positional, dropout_rate=0.1):
        """
        初始化
        :param input_vocab_size: (int, mandatory) 输入词汇大小
        :param output_vocab_size: (int, mandatory) 输出词汇大小
        :param d_model: (int, mandatory) 词嵌入和位置嵌入维度。 通常是 512
        :param heads_num: (int, mandatory) 注意力权重数量。 通常是 8
        :param forward_hidden: (int, mandatory) 隐藏层数量。通常是 2048
        :param num_layers:(int, mandatory) 编码器和解码器的层数量
        :param input_max_positional: (int, mandatory) 输入词汇最大位置编码数量
        :param target_max_positional: (int, mandatory) 输出词汇的最大位置编码数量
        :param dropout_rate: (float, optional, default=0.1) 随机关闭的节点概率
        """
        super(Transformer, self).__init__()

        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.d_model = d_model
        self.heads_num = heads_num
        self.forward_hidden = forward_hidden
        self.num_layers = num_layers
        self.input_max_positional = input_max_positional
        self.target_max_positional = target_max_positional
        self.dropout_rate = dropout_rate

        self.encoder = Encoder(self.input_vocab_size, self.d_model, self.heads_num, self.forward_hidden,
                               self.num_layers, self.input_max_positional, self.dropout_rate)
        self.decoder = Decoder(self.output_vocab_size, self.d_model, self.heads_num, self.forward_hidden,
                               self.num_layers, self.target_max_positional, self.dropout_rate)

        self.final_layer = tf.keras.layers.Dense(self.output_vocab_size)

    def call(self, inputs, decoder_input=None, mask=None, look_ahead_mask=None, padding_mask=None, training=None):
        """
        :param inputs: (tensor, mandatory) 序列数据集。shape: [batch_size, seq_len]
        :param decoder_input: (tensor, mandatory) 目标数据集。shape: [batch_size, seq_len]
        :param mask: (tensor, mandatory) 编码器掩码
        :param look_ahead_mask: (tensor, mandatory) 解码器第一个注意力掩码，前瞻掩码
        :param padding_mask: (tensor, mandatory) 解码器第二个注意力掩码。用于遮挡编码器的输出
        :param training: (bool) 指定是否是训练模式
        :return: (tensor) 计算结果和注意力权重。注意力权重为列表字典格式。
        """
        encoder_output = self.encoder(inputs, mask=mask, training=training)
        decoder_output, attention_weight = self.decoder(decoder_input, encoder_output=encoder_output,
                                                        look_ahead_mask=look_ahead_mask,
                                                        padding_mask=padding_mask,
                                                        training=training)
        output = self.final_layer(decoder_output)
        return output, attention_weight


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    自定义序列时间表。将结合 Adam 优化器来进行自定义调节学习率。根据公式：
    lr = (d_model ** -0.5) * min((step_num ** -0.5), step_num * (warmup_steps ** -1.5))
    论文：https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, warmup_steps=4000, name='warmup_schedule'):
        super(WarmupSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
        self.name = name

    def __call__(self, step, *args, **kwargs):
        lr = (self.d_model ** -0.5) * tf.math.minimum(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return lr

    def get_config(self):
        return {'d_model': self.d_model,
                'warmup_steps': self.warmup_steps,
                'name': self.name}


class LanguageTransformer(object):

    def __init__(self, input_vocab_size, output_vocab_size,
                 d_model=512,
                 heads_num=8,
                 forward_hidden=2048,
                 num_layers=6,
                 input_max_positional=1000,
                 target_max_positional=1000,
                 dropout_rate=0.1,
                 ckpt_dir='./transformer_ckpt',
                 ckpt_max_to_keep=3,
                 lr_warmup_steps=5000,
                 optimizer_beta_1=0.9,
                 optimizer_beta_2=0.98,
                 optimizer_epsilon=1e-9,
                 padding_index=0,
                 pred_max_length=50):
        """
        初始化
        :param input_vocab_size: (int, mandatory) 训练词汇的大小
        :param output_vocab_size: (int, mandatory) 预测词汇的大小
        :param d_model: (int, optional, default=512) 嵌入维度大小
        :param heads_num: (int, optional, default=8) 多头注意力数量
        :param forward_hidden: (int, optional, default=2048) 前馈网络隐藏层大小
        :param num_layers: (int, optional, default=6) 编码层和解码层数量
        :param input_max_positional: (int, optional, default=1000) 输入给编码器的最大位置编码
        :param target_max_positional: (int, optional, default=1000) 输入给解码器的最大位置编码
        :param dropout_rate: (float, optional, default=0.1) 随机关闭节点概率，防止过拟合
        :param ckpt_dir: (str, optional, default='transformer_ckpt') 保存检查点目录
        :param ckpt_max_to_keep: (int, optional, default=3) 保存检查点的数量
        :param lr_warmup_steps: (float, optional, default=5000) 学习率热身函数的 warmup_steps。
        参考：https://arxiv.org/abs/1706.03762
        :param optimizer_beta_1: (float, optional, default=0.9) adam 优化器beta1
        :param optimizer_beta_2: (float, optional, default=0.98) adam 优化器beta2
        :param optimizer_epsilon: (float, optional, default=1e-9) adam 优化器epsilon
        :param padding_index: (int, optional, default=0) 填充字符索引
        :param pred_max_length: (int, optional, default=50) 预测句子的最大长度
        """
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.d_model = d_model
        self.heads_num = heads_num
        self.forward_hidden = forward_hidden
        self.num_layers = num_layers
        self.input_max_positional = input_max_positional
        self.target_max_positional = target_max_positional
        self.dropout_rate = dropout_rate
        self.ckpt_dir = ckpt_dir
        self.ckpt_max_to_keep = ckpt_max_to_keep
        self.lr_warmup_steps = lr_warmup_steps
        self.optimizer_beta_1 = optimizer_beta_1
        self.optimizer_beta_2 = optimizer_beta_2
        self.optimizer_epsilon = optimizer_epsilon
        self.padding_index = padding_index
        self.pred_max_length = pred_max_length

        self.__init_optimizer__()
        self.__init_transformer__()
        self.__init_metrics__()
        self.__init_ckpt__()

        self.minimum_train_loss = None
        self.minimum_valid_loss = None

    def __init_optimizer__(self):
        """
        初始化优化器
        :return:
        """
        lr = WarmupSchedule(d_model=self.d_model, warmup_steps=self.lr_warmup_steps)
        self.optimizer = tf.keras.optimizers.Adam(lr, beta_1=self.optimizer_beta_1, beta_2=self.optimizer_beta_2,
                                                  epsilon=self.optimizer_epsilon)

    def __init_transformer__(self):
        """
        初始化Transformer模型
        :return:
        """
        self.transformer = Transformer(input_vocab_size=self.input_vocab_size,
                                       output_vocab_size=self.output_vocab_size,
                                       d_model=self.d_model,
                                       heads_num=self.heads_num,
                                       forward_hidden=self.forward_hidden,
                                       num_layers=self.num_layers,
                                       input_max_positional=self.input_max_positional,
                                       target_max_positional=self.target_max_positional,
                                       dropout_rate=self.dropout_rate)

    def __init_ckpt__(self):
        """
        初始化检查点
        :return:
        """
        self.checkpoint = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=self.checkpoint, directory=self.ckpt_dir,
                                                             max_to_keep=self.ckpt_max_to_keep)

    def __init_metrics__(self):
        """
        初始化评价指标
        :return:
        """
        # 交叉熵损失。将reduction设置为None， 即关闭自动计算，只获取序列损失。
        # 获取的序列损失还需要再重新计算，忽略填充损失。
        self.sparse_categorical_crossentropy = \
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        self.valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def computer_loss(self, y_true, y_pred):
        """
        由于序列包含填充。在计算损失的时候，需要对填充进行遮挡
        :param y_true: (tensor, mandatory) 正确标签
        :param y_pred: (tensor, mandatory) 预测序列
        :return:
        """
        # 计算损失
        loss = self.sparse_categorical_crossentropy(y_true=y_true, y_pred=y_pred)

        # 获取掩码。即 [[1,1,1,0,0,0]] 格式。 1 代表有数字、0 代表是填充
        mask = tf.math.logical_not(tf.math.equal(y_true, self.padding_index))
        mask = tf.cast(mask, dtype=loss.dtype)

        # 将掩码与loss相乘，即忽略填充损失
        loss = loss * mask

        return tf.reduce_mean(loss)

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

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, inputs, target):
        """
        每个step的训练
        :param inputs: (tensor) 输入编码器的数据
        :param target: (tensor) 输入解码器的数据
        :return:
        """
        # 对目标预测序列进行处理。
        # target_input 为解码器的输入，需要抛弃 <END> 。 在解码器的输入中 <END> 不作为输入
        target_inputs = target[:, :-1]

        # target_real 为实际预测标签，需要抛弃 <START> 。 在预测序列中 <START> 标记不在预测的序列中
        target_real = target[:, 1:]

        # 创建编码器填充掩码 和 解码器掩码(填充掩码和前瞻掩码的合并)、解码器填充掩码(用于对编码器的输出进行遮挡)
        encoder_padding_mask, decoder_mask, decoder_padding_mask = create_mask(inputs, target_inputs,
                                                                               mask_index=self.padding_index)

        with tf.GradientTape() as tape:
            pred, _ = self.transformer(inputs,
                                       decoder_input=target_inputs,
                                       mask=encoder_padding_mask,
                                       look_ahead_mask=decoder_mask,
                                       padding_mask=decoder_padding_mask,
                                       training=True)
            loss = self.computer_loss(target_real, pred)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(target_real, pred)

    @tf.function(experimental_relax_shapes=True)
    def valid_step(self, inputs, target):
        """
        验证
        :param inputs: (tensor) 输入编码器的数据
        :param target: (tensor) 输入解码器的数据
        :return:
        """
        target_inputs = target[:, :-1]
        target_real = target[:, 1:]
        encoder_padding_mask, decoder_mask, decoder_padding_mask = create_mask(inputs, target_inputs,
                                                                               mask_index=self.padding_index)
        pred, _ = self.transformer(inputs,
                                   decoder_input=target_inputs,
                                   mask=encoder_padding_mask,
                                   look_ahead_mask=decoder_mask,
                                   padding_mask=decoder_padding_mask,
                                   training=False)

        loss = self.computer_loss(target_real, pred)
        self.valid_loss(loss)
        self.valid_accuracy(target_real, pred)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, inputs, target):
        """
        验证
        :param inputs: (tensor) 输入编码器的数据
        :param target: (tensor) 输入解码器的数据
        :return:
        """
        target_inputs = target[:, :-1]
        target_real = target[:, 1:]
        encoder_padding_mask, decoder_mask, decoder_padding_mask = create_mask(inputs, target_inputs,
                                                                               mask_index=self.padding_index)
        pred, _ = self.transformer(inputs,
                                   decoder_input=target_inputs,
                                   mask=encoder_padding_mask,
                                   look_ahead_mask=decoder_mask,
                                   padding_mask=decoder_padding_mask,
                                   training=False)

        loss = self.computer_loss(target_real, pred)
        self.test_loss(loss)
        self.test_accuracy(target_real, pred)

    def fit(self, x, y, batch_size=8, epochs=10, validation_split=0.1, shuffle=True):
        """
        训练
        :param x: (list, mandatory) 训练数据集。例如: [[1,2,3],[4,5,6]...]
        :param y: (list, mandatory) 标签数据集。例如：[[1,2,3],[4,5,6]...]
        :param batch_size: (int, optional, default=8) 样本大小
        :param epochs: (int, optional, default=10) 训练轮次
        :param validation_split: (float, optional, default=0.1) 验证数据集比例。0～1范围内
        :param shuffle: (bool, optional, default=True) 是否进行洗牌
        :return: (list) 训练损失、验证损失
        """
        dataset = packaging_dataset(x, y, batch_size=batch_size, shuffle=shuffle)
        train_sample_size, valid_sample_size = get_sample_size(dataset, validation_rate=validation_split)

        print("start fit ...")
        train_loss_save, valid_loss_save = list(), list()
        for e in range(epochs):
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.valid_loss.reset_states()
            self.valid_accuracy.reset_states()

            train_dataset = dataset.skip(valid_sample_size)
            valid_dataset = dataset.take(valid_sample_size)

            # 训练
            for step, (sample_x, sample_y) in enumerate(train_dataset):
                self.train_step(sample_x, sample_y)

                output = template(e + 1, epochs, step + 1, train_sample_size, self.train_loss.result(),
                                  self.train_accuracy.result(), head='Train')
                sys.stdout.write('\r' + output)
                sys.stdout.flush()
            print()
            train_loss_save.append(self.train_loss.result())

            # 验证
            for step, (sample_x, sample_y) in enumerate(valid_dataset):
                self.valid_step(sample_x, sample_y)

                output = template(e + 1, epochs, step + 1, valid_sample_size, self.valid_loss.result(),
                                  self.valid_accuracy.result(), head='Valid')
                sys.stdout.write('\r' + output)
                sys.stdout.flush()
            print()
            valid_loss_save.append(self.valid_loss.result())

            # 保存检查点
            self.train_save_checkpoint()
            print()

    def eval(self, x, y):
        """
        评估。
        :param x: (list, mandatory) 训练数据集。例如: [[1,2,3],[4,5,6]...]
        :param y: (list, mandatory) 标签数据集。例如：[[1,2,3],[4,5,6]...]
        :return: (list) 损失
        """
        print("start eval ...")
        dataset = packaging_dataset(x, y, batch_size=1, shuffle=False)
        test_loss_save = list()

        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        for step, (sample_x, sample_y) in enumerate(dataset):
            self.test_step(sample_x, sample_y)

            output = template(1, 1, step + 1, len(x), self.test_loss.result(), self.test_accuracy.result(),
                              head='Test')
            sys.stdout.write('\r' + output)
            sys.stdout.flush()

            test_loss_save.append(self.test_loss.result())
        print()

        return test_loss_save

    def predict(self, sent, y_start, y_end):
        """
        预测
        :param sent: (list, mandatory) 句子。例如：[1,2,3,4]
        :param y_start: (int, mandatory) <START> 的索引。
        :param y_end: (int, mandatory) <End> 的索引
        :return: (tensor) 输出结果和权重
        """
        sample_x = tf.expand_dims(tf.constant(sent, dtype=tf.int32), axis=0)
        sample_y = tf.expand_dims(tf.constant([y_start], dtype=tf.int32), axis=0)

        weight = None
        output = sample_y
        for _ in range(self.pred_max_length):
            encoder_padding_mask, decoder_mask, decoder_padding_mask = create_mask(sample_x, output,
                                                                                   mask_index=self.padding_index)
            pred, attention_weight = self.transformer(sample_x,
                                                      decoder_input=output,
                                                      mask=encoder_padding_mask,
                                                      look_ahead_mask=decoder_mask,
                                                      padding_mask=decoder_padding_mask,
                                                      training=False)
            weight = attention_weight
            # 选择最后一个序列. shape: [batch_size, 1, vocab_size]
            predictions = pred[:, -1:, :]

            # 使用贪心搜索.
            predictions_id = tf.cast(tf.argmax(predictions, axis=-1), dtype=tf.int32)

            # 如果最后一个是结束标记。则返回结果
            if predictions_id == y_end:
                return tf.squeeze(output, axis=0), weight

            # 将本次预测的结果与上一次的进行拼接。组成新的输入序列
            output = tf.concat((output, predictions_id), axis=-1)

        return tf.squeeze(output, axis=0), weight

    def dump_config(self, file='language_transformer.json'):
        """
        保存配置信息
        :param file: (str, optional, default='language_transformer.json') 配置信息文件名
        :return:
        """
        config = {
            'class': LanguageTransformer.__name__,
            'input_vocab_size': self.input_vocab_size,
            'output_vocab_size': self.output_vocab_size,
            'd_model': self.d_model,
            'heads_num': self.heads_num,
            'forward_hidden': self.forward_hidden,
            'num_layers': self.num_layers,
            'input_max_positional': self.input_max_positional,
            'target_max_positional': self.target_max_positional,
            'dropout_rate': self.dropout_rate,
            'ckpt_dir': self.ckpt_dir,
            'ckpt_max_to_keep': self.ckpt_max_to_keep,
            'lr_warmup_steps': self.lr_warmup_steps,
            'optimizer_beta_1': self.optimizer_beta_1,
            'optimizer_beta_2': self.optimizer_beta_2,
            'optimizer_epsilon': self.optimizer_epsilon,
            'padding_index': self.padding_index,
            'pred_max_length': self.pred_max_length
        }

        Writer.check_path(file)
        with open(file, 'w') as f:
            json.dump(config, f, indent=4)

        print("writer config over! file: {}".format(file))

    @staticmethod
    def from_config(config):
        """
        通过配置还原模型
        :param config: (str, mandatory) 配置文件
        :return: (LanguageTransformer) 对象
        """
        assert os.path.isfile(config), FileNotFoundException(config)
        with open(config, 'r') as f:
            data = json.load(f)
        assert data['class'] == LanguageTransformer.__name__, ParameterError(
            'config:{} class:{} error, expect get{}'.format(config, data['class'], LanguageTransformer.__name__))

        language_transformer = LanguageTransformer(input_vocab_size=data['input_vocab_size'],
                                                   output_vocab_size=data['output_vocab_size'],
                                                   d_model=data['d_model'],
                                                   heads_num=data['heads_num'],
                                                   forward_hidden=data['forward_hidden'],
                                                   num_layers=data['num_layers'],
                                                   input_max_positional=data['input_max_positional'],
                                                   target_max_positional=data['target_max_positional'],
                                                   dropout_rate=data['dropout_rate'],
                                                   ckpt_dir=data['ckpt_dir'],
                                                   ckpt_max_to_keep=data['ckpt_max_to_keep'],
                                                   lr_warmup_steps=data['lr_warmup_steps'],
                                                   optimizer_beta_1=data['optimizer_beta_1'],
                                                   optimizer_beta_2=data['optimizer_beta_2'],
                                                   optimizer_epsilon=data['optimizer_epsilon'],
                                                   padding_index=data['padding_index'],
                                                   pred_max_length=data['pred_max_length'])
        print(Color.green("restore transformer from: {}".format(config)))
        return language_transformer

    @staticmethod
    def restore(config, checkpoint):
        """
        还原模型
        :param config: (str, mandatory) 配置文件
        :param checkpoint: (str) 检查点目录
        :return: (LanguageTransformer) 对象
        """
        language_transformer = LanguageTransformer.from_config(config)

        if tf.train.latest_checkpoint(checkpoint):
            language_transformer.checkpoint.restore(tf.train.latest_checkpoint(checkpoint)).expect_partial()
            output = "restore checkpoint from: {}".format(tf.train.latest_checkpoint(checkpoint))
        else:
            output = "Not found checkpoint Initializing from scratch"

        print(Color.green(output))
        return language_transformer


def test_module_func(optional):
    def virtual_data(sample_sum, sequence_len, padding_size=10, vocab_size=200, d_type=tf.int32):
        sample_data = list()
        for _ in range(sample_sum):
            data = tf.random.uniform((sequence_len,), minval=1, maxval=vocab_size, dtype=d_type)
            data = tf.concat((tf.constant([vocab_size]), data), axis=0)
            data = tf.concat((data, tf.constant([vocab_size + 1])), axis=0)
            data = tf.concat((data, tf.constant([0] * padding_size)), axis=0)
            sample_data.append(data)

        return sample_data

    if optional == 'positional_encoding':
        output = positional_encoding(50, 512)
        assert output.shape == (1, 50, 512)
        print("positional encoding: ", output.shape)
        show_positional_encoding(output[0])

    elif optional == 'padding_mask':
        seq = tf.constant([[1, 2, 3, 0], [2, 3, 4, 0], [1, 1, 1, 0]])
        true_return = tf.constant([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], dtype=tf.float32)
        output = create_padding_mask(seq, mask_index=0)
        assert tf.reduce_all(tf.cast(output, tf.bool)) == tf.reduce_all(tf.cast(true_return, tf.bool))
        assert output.shape == (3, 1, 1, 4)
        print(output)

    elif optional == "ahead_mask":
        size = 3
        true_return = tf.constant([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        output = create_look_ahead_mask(size)
        assert tf.reduce_all(tf.cast(output, tf.bool)) == tf.reduce_all(tf.cast(true_return, tf.bool))
        assert output.shape == (3, 3)
        print(output)

    elif optional == 'scaled_dot_attention':
        temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)
        temp_k = tf.constant([[10, 0, 0],
                              [0, 10, 0],
                              [0, 0, 10],
                              [0, 0, 10]], dtype=tf.float32)
        temp_v = tf.constant([[1, 0],
                              [10, 0],
                              [100, 5],
                              [1000, 6]], dtype=tf.float32)

        output, attention_weight = scaled_dot_product_attention(temp_q, temp_k, temp_v, mask=None)
        print("q: ", temp_q)
        print("scaled dot:", output)
        print("attention weight: ", attention_weight)

        temp_q = tf.constant([[0, 10, 10]], dtype=tf.float32)
        output, attention_weight = scaled_dot_product_attention(temp_q, temp_k, temp_v, mask=None)
        print("q: ", temp_q)
        print("scaled dot:", output)
        print("attention weight: ", attention_weight)

    elif optional == 'multi_head_attention':
        batch_size = 64
        seq_len = 60
        d_model = 512

        multi_head_attention = MultiHeadAttention(d_model=d_model, head_num=8)
        q = tf.random.uniform((batch_size, seq_len, d_model))

        k = q
        v = q
        mask = None
        output, attention_weight = multi_head_attention((q, k, v), mask=mask)
        print("output shape: ", output.shape)
        print("attention weight: ", attention_weight.shape)

    elif optional == 'feed_forward_network':
        feed_forward = point_wise_feed_forward_network(2048, 512)
        output = feed_forward(tf.random.uniform((64, 50, 512)))
        assert output.shape == (64, 50, 512)
        print("feed forward network: ", output.shape)

    elif optional == 'encoder_layer':
        batch_size = 64
        seq_len = 50
        d_model = 512
        encoder = EncoderLayer(d_model=d_model, heads_num=8, forward_hidden=2048)
        output = encoder(tf.random.uniform((batch_size, seq_len, d_model)), mask=None, training=False)
        assert output.shape == (batch_size, seq_len, d_model)

        print("encoder layer: ", output.shape)

    elif optional == 'decoder_layer':
        batch_size = 64
        seq_len = 50
        d_model = 512
        encoder = EncoderLayer(d_model=d_model, heads_num=8, forward_hidden=2048)
        encoder_output = encoder(tf.random.uniform((batch_size, seq_len, d_model)), mask=None, training=False)

        decoder_layer = DecoderLayer(d_model=d_model, heads_num=8, forward_hidden=2048)
        inputs = tf.random.uniform((batch_size, seq_len, d_model))
        decoder_output, _, _ = decoder_layer(inputs, encoder_output=encoder_output, look_ahead_mask=None,
                                             padding_mask=None,
                                             training=False)

        assert decoder_output.shape == (batch_size, seq_len, d_model)
        print("decoder output: ", decoder_output.shape)

    elif optional == 'encoder':
        encoder = Encoder(vocab_size=10240, d_model=512, heads_num=8, forward_hidden=2048, num_layers=2,
                          max_positional_encoding=256)

        inputs = tf.random.uniform((64, 256), minval=0, maxval=10240, dtype=tf.int32)
        output = encoder(inputs, mask=None, training=False)

        print("encoder shape: ", output.shape)

    elif optional == 'decoder':
        inputs = tf.random.uniform((64, 256), minval=0, maxval=10240, dtype=tf.int32)

        encoder = Encoder(vocab_size=10240, d_model=512, heads_num=8, forward_hidden=2048, num_layers=2,
                          max_positional_encoding=256)
        decoder = Decoder(vocab_size=10240, d_model=512, heads_num=8, forward_hidden=2048, num_layers=2,
                          max_positional_encoding=256)

        encoder_output = encoder(inputs, mask=None, training=False)
        output, attention_weight = decoder(inputs, encoder_output=encoder_output, look_ahead_mask=None,
                                           padding_mask=None, training=False)

        print("decoder shape: ", output.shape)
        print("attention weight: ", attention_weight[0]['decoder_layer0_attention_weight1'].shape)

    elif optional == 'transformer':
        transformer = Transformer(input_vocab_size=6000, output_vocab_size=8000,
                                  d_model=512, heads_num=8, forward_hidden=512, num_layers=3,
                                  input_max_positional=1000, target_max_positional=600, dropout_rate=0.1)

        encoder_inputs = tf.random.uniform((64, 256), minval=0, maxval=6000, dtype=tf.int32)
        decoder_inputs = tf.random.uniform((64, 128), minval=0, maxval=8000, dtype=tf.int32)
        output, attention_weight = transformer(encoder_inputs, decoder_inputs, mask=None, look_ahead_mask=None,
                                               padding_mask=None, training=False)
        print("transformer output: ", output.shape)
        print("attention weight: ", attention_weight[0]['decoder_layer0_attention_weight1'].shape)

    elif optional == 'warmup_schedule':
        warmup_schedule = WarmupSchedule(d_model=512, warmup_steps=1000)
        for e in range(10):
            for step in range(80):
                output = warmup_schedule(step + 1)
                print("e: {}, step: {}, output: {}".format(e, step, output))
            print()
        # plt.plot(warmup_schedule(tf.range(40000, dtype=tf.float32)))
        # plt.xlabel("Train steps")
        # plt.ylabel("Learning Rate")
        # plt.show()

    elif optional == 'packaging_dataset':
        x, y = list(), list()
        for i in range(100):
            x.append(np.arange(10) * (i + 1))
            y.append(np.arange(10) * (i + 2))

        dataset = packaging_dataset(x, y, batch_size=4, shuffle=True)
        for i, (x, y) in enumerate(dataset):
            print("sample: ", i + 1)
            print("x:", x)
            print("y:", y)
            print()

    elif optional == 'get_sample_size':
        x, y = list(), list()
        for i in range(100):
            x.append(np.arange(10) * (i + 1))
            y.append(np.arange(10) * (i + 2))
        dataset = packaging_dataset(x, y, batch_size=4, shuffle=True)

        train_size, valid_size = get_sample_size(dataset, validation_rate=0.2)
        print("train sample size: ", train_size)
        print("valid sample size: ", valid_size)

        train_dataset = dataset.skip(valid_size)
        valid_dataset = dataset.take(valid_size)
        print("train dataset size: ", len(list(train_dataset)))
        print("valid dataset size: ", len(list(valid_dataset)))

    elif optional == 'language_transformer_fit':
        sample_size = 100
        x_vocab_size = 200
        y_vocab_size = 220
        x = virtual_data(sample_size, 32, 5, x_vocab_size)
        y = virtual_data(sample_size, 36, 6, y_vocab_size)

        model = LanguageTransformer(input_vocab_size=x_vocab_size + 2, output_vocab_size=y_vocab_size + 2)
        model.fit(x, y, epochs=2)
        # conf = model.optimizer.get_config()
        # print(conf)

    elif optional == 'language_transformer_eval':
        sample_size = 100
        x_vocab_size = 200
        y_vocab_size = 220
        x = virtual_data(sample_size, 32, 5, x_vocab_size)
        y = virtual_data(sample_size, 36, 6, y_vocab_size)

        model = LanguageTransformer(input_vocab_size=x_vocab_size + 2, output_vocab_size=y_vocab_size + 2)
        model.eval(x, y)

    elif optional == 'language_transformer_pred':
        sample_size = 1
        x_vocab_size = 200
        y_vocab_size = 220
        x = virtual_data(sample_size, 32, 5, x_vocab_size)

        y_start, y_end = 220, 221
        model = LanguageTransformer(input_vocab_size=x_vocab_size + 2, output_vocab_size=y_vocab_size + 2)
        output, weight = model.predict(x[0], y_start, y_end)
        print("predict output: ", output)

    elif optional == 'language_transformer_dump_restore':
        ckpt = './ckpt'
        config = './language_transformer.json'
        model = LanguageTransformer(input_vocab_size=220, output_vocab_size=240, ckpt_dir=ckpt)
        model.dump_config(config)
        model.checkpoint_manager.save()

        new_model = LanguageTransformer.restore(config, ckpt)

        sample_size = 1
        x_vocab_size = 200
        x = virtual_data(sample_size, 32, 5, x_vocab_size)

        y_start, y_end = 220, 221
        output, weight = new_model.predict(x[0], y_start, y_end)
        print("restore model predict output: ", output)


def main():
    # test_module_func('positional_encoding')
    # test_module_func('padding_mask')
    # test_module_func('ahead_mask')
    # test_module_func('scaled_dot_attention')
    # test_module_func('multi_head_attention')
    # test_module_func('feed_forward_network')
    # test_module_func('encoder_layer')
    # test_module_func('decoder_layer')
    # test_module_func('encoder')
    # test_module_func('decoder')
    # test_module_func('transformer')
    # test_module_func('warmup_schedule')
    # test_module_func('packaging_dataset')
    # test_module_func('get_sample_size')
    test_module_func('language_transformer_fit')
    # test_module_func('language_transformer_eval')
    # test_module_func('language_transformer_pred')
    # test_module_func('language_transformer_dump_restore')


if __name__ == '__main__':
    main()
