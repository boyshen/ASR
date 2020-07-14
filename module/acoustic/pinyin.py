# -*- encoding: utf-8 -*-
"""
@file: pinyin.py
@time: 2020/5/23 下午4:48
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: 
"""

import numpy as np
from tqdm import tqdm
from collections import Counter
from module.core.utils import Dictionary
from module.core.exception import exception_handling, NotFitException


class PinYin(Dictionary):

    def __init__(self):
        super(PinYin, self).__init__()
        self.SPACE_TAG = '<SPACE>'
        self.SPACE = 0

        self.pinyin_token = dict()
        self.token_pinyin = dict()

        # 保存拟合语料中最长序列
        self.max_length = 0

        self.is_fit = False

    def fit(self, dataset):
        """
        初始化数据词典
        :param dataset: (list, mandatory) 数据列表。例如[['a','b','c','d'...]...]
        :return:
        """
        pinyin_counter = Counter()

        for data in tqdm(dataset):
            pinyin_counter.update(data)
            if len(data) > self.max_length:
                self.max_length = len(data)

        for pinyin, count in pinyin_counter.items():
            self.pinyin_token[pinyin] = len(self.pinyin_token)
        self.pinyin_token[self.SPACE_TAG] = len(self.pinyin_token)
        self.SPACE = self.pinyin_token[self.SPACE_TAG]

        self.token_pinyin = {token: pinyin for pinyin, token in self.pinyin_token.items()}

        self.is_fit = True

        print("fit over ! pinyin: {}, Max length: {}".format(len(self.pinyin_token), self.max_length))

    def __len__(self):
        return len(self.pinyin_token)

    @exception_handling
    def pinyin_to_token(self, pinyin):
        """
        将拼音转换成 token
        :param pinyin: (str, mandatory) 拼音
        :return: (int) token 或 UNK=0
        """
        assert self.is_fit, NotFitException(PinYin.__name__)
        try:
            return self.pinyin_token[pinyin]
        except KeyError:
            return self.SPACE

    @exception_handling
    def token_to_pinyin(self, token):
        """
        将 token 转换成拼音
        :param token: (int, mandatory) token 标识符
        :return: (str) word 或 <UNK>
        """
        assert self.is_fit, NotFitException(PinYin.__name__)
        try:
            return self.token_pinyin[token]
        except KeyError:
            return self.SPACE_TAG

    def sequence_alignment(self, seq, alignment_size=None):
        """
        序列对齐。指定输出大小相同的序列。如果 len(seq) > alignment_size 。则进行序列截断。小于则使用 SPACE 填充
        :param seq: (list, mandatory) 句子序列。例如: ['a','b','c','d']
        :param alignment_size: (int, optional, default=None) 对齐大小。如果为None, 则使用拟合语料中得到的 self.max_length
        :return: (array) 填充序列, 序列大小
        """
        sequence = [self.pinyin_to_token(pinyin) for pinyin in seq]
        alignment_size = self.max_length if alignment_size is None else alignment_size

        padding_sequence = np.ones(alignment_size) * self.SPACE

        # 如果句子序列大于对齐大小。则进行截断
        if len(sequence) > alignment_size:
            sequence = sequence[:alignment_size]

        padding_sequence[:len(sequence)] = sequence

        return padding_sequence, len(sequence)

    def encoding(self, seq, alignment=False, alignment_size=None):
        """
        将拼音列表转换成token列表
        :param seq: (list, mandatory) 拼音列表。例如['a','b','c','d']
        :param alignment: (bool, optional, default=True) 是否对齐
        :param alignment_size: (int, optional, default=None) 对齐大小
        :return: (array) 例如：[1,2,3,4]
        """
        if alignment:
            sequence, sequence_length = self.sequence_alignment(seq, alignment_size)
        else:
            sequence = np.array([self.pinyin_to_token(pinyin) for pinyin in seq])
            sequence_length = len(sequence)
        return sequence, sequence_length

    def decoding(self, seq):
        """
        将数字列表转换成拼音列表
        :param seq: (list, mandatory) 数字列表。例如[1,2,3,4]
        :return: (list) 。拼音
        """
        sequence = []
        for token in seq:
            if token == self.SPACE:
                continue
            sequence.append(self.token_to_pinyin(token))

        return sequence


def test_module_func():
    dataset = [['a', 'b', 'c', 'd', 'e'], ['a', 's', 'd', 'f']]

    pinyin = PinYin()
    pinyin.fit(dataset)

    assert len(pinyin) == 7 + 1

    print(pinyin.pinyin_token)
    print(pinyin.token_pinyin)

    token = pinyin.pinyin_to_token('a')
    assert pinyin.token_to_pinyin(token) == 'a'

    token = pinyin.pinyin_to_token('as')
    assert pinyin.token_to_pinyin(token) == pinyin.SPACE_TAG
    assert pinyin.pinyin_to_token(pinyin.SPACE_TAG) == 7

    encoding_token, encoding_length = pinyin.encoding(['a', 'b', 'c', 'd'], alignment=False)
    assert (encoding_token == np.array([0, 1, 2, 3])).all()
    assert encoding_length == 4

    encoding_token, encoding_length = pinyin.encoding(['a', 'b', 'c', 'd'], alignment=True)
    assert (encoding_token == np.array([0, 1, 2, 3, 7])).all()
    assert encoding_length == 4

    encoding_token, encoding_length = pinyin.encoding(['a', 'b', 'c', 'd'], alignment=True, alignment_size=6)
    assert (encoding_token == np.array([0, 1, 2, 3, 7, 7])).all()
    assert encoding_length == 4

    encoding_token, encoding_length = pinyin.encoding(['a', 'b', 'c', 'd', 'e', 's'], alignment=True)
    assert (encoding_token == np.array([0, 1, 2, 3, 4])).all()
    assert encoding_length == 5

    decoding_str = pinyin.decoding([0, 1, 2, 3, 7, 7, 7])
    assert decoding_str == ['a', 'b', 'c', 'd']

    print("test ok !")


if __name__ == "__main__":
    test_module_func()
