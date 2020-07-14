# -*- encoding: utf-8 -*-
"""
@file: char_map.py
@time: 2020/6/1 下午4:48
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: 字符映射。将每一个字符映射成一个唯一数字token。例如 'a' -> 1, 'b' -> 2, 'c' -> 3
"""
import re
from tqdm import tqdm
from collections import Counter
from module.core.utils import Dictionary
from module.core.exception import NotFitException


class CharMap(Dictionary):
    BLACK_TAG = '<BLACK>'
    SPACE_TAG = '<SPACE>'

    # BLACK = 0
    SPACE = 0

    def __init__(self):
        super(CharMap, self).__init__()

        self.char_token = {
            CharMap.SPACE_TAG: CharMap.SPACE
        }
        self.token_char = {}

        self.max_length = 0

        self.is_fit = False

    def __len__(self):
        return len(self.char_token)

    def fit(self, dataset):
        """
        拟合。创建字符映射。
        :param dataset: (list, mandatory) 数据集。格式：[['ni1','hao2','ma3'...]]
        :return:
        """
        char_counter = Counter()

        for data in tqdm(dataset):
            length = 0
            for word in data:
                # 处理空格字符。发现空格字符则跳过。
                word = re.sub('[ ]+', '', word)
                if word == ' ' or word == '' or len(word) == 0:
                    continue
                char_counter.update(list(word))

                # 保存标签语料的长度， 1 是指 <SPACE> 每个词后面有一个空格字符。
                length += len(list(word)) + 1

            # 对比保存最大的标签语料。 1 是指最后的一个词没 <SPACE> 字符
            if self.max_length < length - 1:
                self.max_length = length - 1

        for char, _ in char_counter.items():
            self.char_token[char] = len(self.char_token)
        self.char_token[CharMap.BLACK_TAG] = len(self.char_token)

        self.token_char = {token: char for char, token in self.char_token.items()}

        self.is_fit = True
        print("char_map fit over ! get char number: {}, max length: {}".format(len(self.char_token), self.max_length))

    def char_to_token(self, char):
        """
        将字符转换成 token
        :param char: (str, mandatory) 字符。例如 'a'
        :return:  (int) token
        """
        assert self.is_fit, NotFitException("{} not fit".format(CharMap.__name__))
        return self.char_token[char]

    def token_to_char(self, token):
        """
        将token还原成字符
        :param token: (int, mandatory) token
        :return: (str) 字符
        """
        assert self.is_fit, NotFitException("{} not fit".format(CharMap.__name__))
        return self.token_char[token]

    def string_to_token(self, string):
        """
        将字符串转换成token。 例如输入: 'hello' -> [3,4,5,5,8]
        :param string: (str, mandatory) 字符串
        :return: (list) token
        """
        return [self.char_to_token(char) for char in list(string)]

    def token_to_string(self, token):
        """
        将 token 转换成字符串。例如输入：[3,4,5,5,8] -> ['h','e','l','l','o']
        :param token: (int, mandatory) token
        :return: (list) 字符列表
        """
        return [self.token_to_char(t) for t in token]

    def encoding(self, char_sequence):
        """
        将句子编码成token列表的格式。 例如输入：['hello', 'world'] -> [3,4,5,5,8,10,2,8,7,5,9] 其中 10 为 <SPACE> 空白符
        :param char_sequence: (list, mandatory) 句子列表。例如: ['hello', 'word']
        :return: (list) token
        """
        int_sequence = list()
        for word in char_sequence:
            if word == '' or word == ' ' or len(word) == 0:
                continue
            if word == char_sequence[-1]:
                int_sequence = int_sequence + self.string_to_token(word)
            else:
                int_sequence = int_sequence + self.string_to_token(word) + [self.char_to_token(CharMap.SPACE_TAG)]

        return int_sequence

    def decoding(self, int_sequence):
        """
        将数字序列还原成字符串列表。例如输入：[3,4,5,5,8,10,2,8,7,5,9] 输出 ['hello', 'world']
        :param int_sequence: (list, mandatory) 数字列表
        :return: (list) 字符列表
        """
        new_sequence = list()
        for token in int_sequence:
            if token == self.char_to_token(CharMap.BLACK_TAG):
                continue
            new_sequence.append(token)

        char_sequence = self.token_to_string(new_sequence)
        string = ''.join(char_sequence)
        word_list = string.split(CharMap.SPACE_TAG)

        sent = list()
        for word in word_list:
            if word == '' or word == ' ' or len(word) == 0:
                continue
            sent.append(word)

        return sent


def test_module_func():
    dataset = 'lv4 shi4 yang2'.split(' ')

    char_map = CharMap()
    char_map.fit([dataset])

    assert len(char_map) == 13
    assert char_map.max_length == 14

    assert char_map.char_to_token(CharMap.BLACK_TAG) == 12
    assert char_map.char_to_token(CharMap.SPACE_TAG) == 0
    assert char_map.token_to_char(12) == CharMap.BLACK_TAG
    assert char_map.token_to_char(0) == CharMap.SPACE_TAG

    char_v_token = char_map.char_to_token('v')
    assert char_map.token_to_char(char_v_token) == 'v'

    char = char_map.token_to_char(8)
    assert char_map.char_to_token(char) == 8

    char_sequence = ['lv4', 'shi4']
    assert char_map.encoding(char_sequence) == [1, 2, 3, 0, 4, 5, 6, 3], \
        "char:{} encoding:{}".format(char_sequence, char_map.encoding(char_sequence))

    int_sequence = [1, 2, 3, 0, 4, 5, 6, 3, 12, 12, 12]
    assert char_map.decoding(int_sequence) == char_sequence, \
        "int:{} decoding:{}".format(int_sequence, char_map.decoding(int_sequence))

    int_sequence = [1, 2, 3, 0, 4, 5, 6, 3]
    assert char_map.decoding(int_sequence) == char_sequence, \
        "int:{} decoding:{}".format(int_sequence, char_map.decoding(int_sequence))

    print("test ok !")
    print(char_map.char_token)


if __name__ == "__main__":
    test_module_func()
