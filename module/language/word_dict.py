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


class WordDict(Dictionary):

    def __init__(self):
        super(WordDict, self).__init__()
        self.PAD = 0
        self.START = None
        self.END = None

        self.word_token = {
            self.PAD_TAG: self.PAD,
        }
        self.token_word = dict()
        self.is_fit = False

        # 保存拟合数据中，序列长度最大的。用在encoding中对数据进行对齐。
        self.max_seq_value = 0

    def fit(self, dataset):
        """
        初始化数据词典
        :param dataset: (list, mandatory) 数据列表。例如[['a','b','c','d'...]...]
        :return:
        """
        word_counter = Counter()

        for data in tqdm(dataset):
            word_counter.update(data)
            if self.max_seq_value < len(data):
                self.max_seq_value = len(data)
        # 添加上 <START> 和 <END>
        self.max_seq_value += 2

        for word, count in word_counter.items():
            self.word_token[word] = len(self.word_token)
        self.word_token[self.START_TAG] = len(self.word_token)
        self.word_token[self.END_TAG] = len(self.word_token)

        self.START = self.word_token[self.START_TAG]
        self.END = self.word_token[self.END_TAG]

        self.token_word = {token: word for word, token in self.word_token.items()}

        self.is_fit = True
        print("fit over ! words: {}".format(len(self.word_token)))

    def __len__(self):
        """计算词数量大小。不包括填充0"""
        return len(self.word_token) - 0

    @exception_handling
    def word_to_token(self, word):
        """
        将 word 转换成 token
        :param word: (str, mandatory) 单词
        :return: (int) token 或 UNK=0
        """
        assert self.is_fit, NotFitException(WordDict.__name__)
        try:
            return self.word_token[word]
        except KeyError:
            return self.PAD

    @exception_handling
    def token_to_word(self, token):
        """
        将 token 转换成 word
        :param token: (int, mandatory) token 标识符
        :return: (str) word 或 <UNK>
        """
        assert self.is_fit, NotFitException(WordDict.__name__)
        try:
            return self.token_word[token]
        except KeyError:
            return self.PAD_TAG

    def sequence_alignment(self, sent):
        """
        序列对齐
        :param sent: (array, mandatory) 序列
        :return: (array) 返回序列大小 = self.max_seq_value 的序列
        """
        seq = [self.word_to_token(word) for word in sent]
        if len(seq) + 2 > self.max_seq_value:
            seq = [self.START] + seq[:self.max_seq_value - 2] + [self.END]
        else:
            seq = [self.START] + seq + [self.END]
            padding_seq = np.array([self.PAD] * self.max_seq_value)
            padding_seq[:len(seq)] = seq
            seq = padding_seq

        return seq

    def encoding(self, sent, alignment=False):
        """
        将字符列表转换成token列表
        :param sent: (list, mandatory) 字符列表。例如:['a','b','c','d']
        :param alignment: (bool, optional, default=False) 是否对齐数据。
        如果为 True，则大于 self.max_seq_value 的数据将进行截断，小于 self.max_seq_value 的数据则使用PAD填充
        :return: (array) 例如：[5,1,2,3,4,6] 其中 5、6 为 <start> 和 <end>
        """
        if alignment:
            seq = self.sequence_alignment(sent)
        else:
            seq = [self.START] + [self.word_to_token(word) for word in sent] + [self.END]

        return seq

    def decoding(self, tokens):
        """
        将token列表还原成字符列表
        :param tokens: (list, mandatory) token 列表。 例如：[1,2,3,4]
        如果序列中包含 PAD 填充码，当 return_padding = True 时，返回包含填充码的序列，为 False 时返回没有填充码的序列。
        :return: (list) 例如:['a','b','c','d']
        """
        seq = list()
        for token in tokens:
            if token == self.START:
                continue
            if token == self.END:
                break
            seq.append(self.token_to_word(token))
        return seq


def test_module_func():
    dataset = [['a', 'b', 'c', 'd', 'e'], ['a', 's', 'd', 'f']]

    word_dict = WordDict()
    word_dict.fit(dataset)

    assert len(word_dict) == 7 + 3
    assert word_dict.max_seq_value == 7
    assert word_dict.PAD == 0
    assert word_dict.word_to_token(word_dict.PAD_TAG) == word_dict.PAD
    assert word_dict.token_to_word(word_dict.PAD) == word_dict.PAD_TAG

    print(word_dict.word_token)
    print(word_dict.token_word)

    token = word_dict.word_to_token('a')
    assert word_dict.token_to_word(token) == 'a'

    token = word_dict.word_to_token('as')
    assert word_dict.token_to_word(token) == WordDict.PAD_TAG

    encoding_token = word_dict.encoding(['a', 'b', 'c', 'z'])
    assert (encoding_token == np.array([8, 1, 2, 3, 0, 9])).all()

    encoding_token = word_dict.encoding(['a', 'b', 'c', 'd'], alignment=True)
    assert (encoding_token == np.array([8, 1, 2, 3, 4, 9, 0])).all()

    encoding_token = word_dict.encoding(['a', 'b', 'c', 'd', 'e', 's'], alignment=True)
    assert (encoding_token == np.array([8, 1, 2, 3, 4, 5, 9])).all()

    decoding_str = word_dict.decoding([8, 1, 2, 3, 4, 9, 0])
    assert decoding_str == ['a', 'b', 'c', 'd']

    print("test ok !")


if __name__ == "__main__":
    test_module_func()
