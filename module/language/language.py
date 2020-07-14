# -*- encoding: utf-8 -*-
"""
@file: language.py
@time: 2020/6/29 下午5:20
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 训练: python language.py train
# 测试：python language.py test
# 预测: python language.py prediction --sentence [str]
    --sentence: 指定预测的拼音字符串

example:
    python language.py prediction --sentence "lv4 shi4 yang2 chun1 yan1 jing3 da4 kuai4 wen2 zhang1"
"""

import os
import re
import argparse
import numpy as np
from tqdm import tqdm

try:
    from module.core.exception import FileNotFoundException
    from module.core.utils import DataUtils
    from module.language.word_dict import WordDict
    from module.core.color import Color
    from module.core.utils import Config
    from module.language.transformer import LanguageTransformer
except ModuleNotFoundError:
    import sys

    sys.path.append('../../')
    from module.core.exception import FileNotFoundException
    from module.core.utils import DataUtils
    from module.language.word_dict import WordDict
    from module.core.color import Color
    from module.core.utils import Config
    from module.language.transformer import LanguageTransformer

CONFIG_PATH = './config.json'

M_TRAIN = 'train'
M_TEST = 'test'
M_PREDICTION = 'prediction'

MODE = (M_TRAIN, M_TEST, M_PREDICTION)


def input_args():
    parser = argparse.ArgumentParser(description=" 1.train  2.test  3.prediction ")
    parser.add_argument("mode", metavar='operation mode', choices=MODE, help="Select mode:{}".format(MODE))
    parser.add_argument("--sentence", dest='sentence', action='store', type=str, default='',
                        help="Please input prediction sentence !")
    args = parser.parse_args()
    return args


def regular(sent):
    """
    正则化函数
    :param sent: (str, mandatory) 处理拼音句子。例如 'lv4 shi4 yang2'
    :return: (list) 字符列表。 例如：['lv4','shi4','yang2']
    """
    sent = sent.lower()
    return re.findall('[a-z0-9]+', sent)


def handle(sent):
    """
    处理每行文本的函数
    :param sent: (str, mandatory) 字符／句子。例如 hello word
    :return: (list) 列表和原句子。例如 [hello, world], hello world
    """
    line_data = list()
    for word in sent.split(' '):
        # 处理空格字符。发现空格字符则跳过。
        word = re.sub('[ ]+', '', word)
        if word == ' ' or word == '' or len(word) == 0:
            continue
        line_data.append(word)
    return line_data


def load_dataset(source_data_path, data_path):
    """
    加载数据集
    :param data_path: (str, mandatory) 训练数据所在的路径
    :param source_data_path: (str, mandatory) 实际数据路径
    :return: (dict of list) 列表字典。返回音频文件路径，标签文件路径，标签数据路径，拼音字符，汉字
    """

    pinyin_sent_data = list()
    chinese_sent_data = list()

    chinese_data = list()
    pinyin_data = list()

    data_file = os.listdir(data_path)
    for file in tqdm(data_file):
        # 检查文件名后缀是不是 wav 音频类型文件。
        if file[-3:] == 'wav':
            # 标签文件。根据音频文件名得到标签文件名
            label_file = file + '.trn'

            # 判断标签文件是否存在。不存在则报错
            assert label_file in data_file, FileNotFoundException(label_file)

            # 音频文件路径和标签文件路径
            label_file_path = os.path.join(data_path, label_file)

            # 读取标签文件内容。找到对应标签数据路径
            label_data_file_path = DataUtils.read_text_data(label_file_path, show_progress_bar=False)
            assert len(label_data_file_path) == 1, \
                'Get redundant data path: {}, label_file_path:{}'.format(label_data_file_path, label_file_path)

            # 重新拼接路径
            label_data_file_path = os.path.join(source_data_path, label_data_file_path[0].split('/')[-1])
            assert os.path.isfile(label_data_file_path), FileNotFoundException(label_data_file_path)

            # 读取标签数据。包括拼音、文本
            text_data = DataUtils.read_text_data(label_data_file_path, handle_func=handle, show_progress_bar=False)
            chinese = text_data[0]
            pinyin = text_data[1]

            # 检查是否存在重复的数据，如果存在则跳过
            chinese_sent = ''.join(chinese)
            pinyin_sent = ' '.join(pinyin)
            if chinese_sent in chinese_sent_data and pinyin_sent in pinyin_sent_data:
                # print("repeat sent: {}, {}".format(chinese_sent, pinyin_sent))
                continue
            else:
                chinese_sent_data.append(chinese_sent)
                pinyin_sent_data.append(pinyin_sent)

            # 检查pinyin字符是否正确。主要判断每个拼音的最后一个字符是不是数字。不是数字则报错
            for py in pinyin:
                assert py[-1].isdigit(), "the last character:{} of Pinyin is not a number! " \
                                         "pinyin_str:{}, " \
                                         "label_data_file_path:{}, " \
                                         "label_file_path:{}".format(py, pinyin, label_data_file_path,
                                                                     label_file_path)

            # 将由多个中文字符组成的词转换成单个字
            new_chinese = list()
            for ch in chinese:
                new_chinese += list(ch)
            chinese = new_chinese

            # 检查是否是一个拼音对应中文字符， 如果不是则报错
            assert len(chinese) == len(pinyin), "the number of pinyin:{} and chinese:{} is different, " \
                                                "chinese:{}, pinyin:{}, file:{}".format(len(pinyin), len(chinese),
                                                                                        chinese, pinyin,
                                                                                        label_data_file_path)
            chinese_data.append(chinese)
            pinyin_data.append(pinyin)

    print(Color.red("load to {} of data!".format(len(chinese_data))))
    return chinese_data, pinyin_data


def check_input_sentence(sent):
    """
    检查输入的pinyin句子是否满足要求
    :param sent: (str, mandatory) 拼音句子。例如 'lv4 shi4 yang2'
    :return: (bool) True or False
    """
    if sent is None:
        print(Color.red("Please input the prediction sentence !"))
        return False

    sent = regular(sent)
    if len(sent) == 0:
        print(Color.red("Please input the prediction sentence ! sentence cannot be empty!"))
        return False

    return True


def init_or_restore_pinyin_dict(dataset, file):
    """
    初始化或还原拼音字典
    :param dataset:  (list, mandatory) 数据集
    :param file: (str, mandatory) 保存或者还原pinyin字典的文件
    :return: (WordDict) 字典对象
    """
    if os.path.isfile(file):
        pinyin = WordDict.load(file)
        output = 'Restore pinyin dict from file:{}'.format(file)
    else:
        pinyin = WordDict()
        pinyin.fit(dataset=dataset)
        pinyin.save(file)
        output = 'Initialize pinyin dict form dataset'

    print(Color.green(output))
    return pinyin


def init_or_restore_chinese_dict(dataset, file):
    """
    初始化或还原中文字典
    :param dataset: (list, mandatory) 数据集
    :param file: (str, mandatory) 保存或者还原chinese的文件
    :return: (WordDict) 字典对象
    """
    if os.path.isfile(file):
        chinese = WordDict.load(file)
        output = 'Restore chinese dict from file: {}'.format(file)
    else:
        chinese = WordDict()
        chinese.fit(dataset=dataset)
        chinese.save(file)
        output = 'Initialize chinese dict from dataset'
    print(Color.green(output))
    return chinese


def restore_transformer(file, ckpt_dir):
    """
    还原 transformer
    :param file: (str, mandatory) transformer 配置文件
    :param ckpt_dir: (str, mandatory) 检查点保存目录
    :return: (LanguageTransformer) transformer 对象
    """
    try:
        language_transformer = LanguageTransformer.restore(file, ckpt_dir)
    except Exception as e:
        print(Color.red("restore transformer model failed！config file: {}, checkpoint: {}".format(file, ckpt_dir)))
        print(e)
        return False

    return language_transformer


def init_transformer(input_vocab_size, output_vocab_size, padding_index, conf):
    """
    初始化 transformer
    :param input_vocab_size: (int, mandatory) 输入词汇大小
    :param output_vocab_size: (int, mandatory) 输出词汇大小
    :param padding_index: (int, mandatory) 填充字符索引
    :param conf: (object) 配置信息
    :return: (LanguageTransformer) transformer 对象
    """
    transformer = LanguageTransformer(input_vocab_size=input_vocab_size,
                                      output_vocab_size=output_vocab_size,
                                      d_model=conf.MODEL.TRANSFORMER.d_model,
                                      heads_num=conf.MODEL.TRANSFORMER.heads_num,
                                      forward_hidden=conf.MODEL.TRANSFORMER.forward_hidden,
                                      num_layers=conf.MODEL.TRANSFORMER.num_layers,
                                      input_max_positional=conf.MODEL.TRANSFORMER.input_max_positional,
                                      target_max_positional=conf.MODEL.TRANSFORMER.target_max_positional,
                                      dropout_rate=conf.MODEL.TRANSFORMER.dropout_rate,
                                      ckpt_dir=conf.MODEL.TRANSFORMER.ckpt_dir,
                                      ckpt_max_to_keep=conf.MODEL.TRANSFORMER.ckpt_max_to_keep,
                                      lr_warmup_steps=conf.MODEL.TRANSFORMER.lr_warmup_steps,
                                      optimizer_beta_1=conf.MODEL.TRANSFORMER.optimizer_beta_1,
                                      optimizer_beta_2=conf.MODEL.TRANSFORMER.optimizer_beta_2,
                                      optimizer_epsilon=conf.MODEL.TRANSFORMER.optimizer_epsilon,
                                      padding_index=padding_index,
                                      pred_max_length=conf.MODEL.TRANSFORMER.pred_max_length)

    print(Color.green('Initialization transformer from scratch'))
    return transformer


def init_or_restore_transformer(file, ckpt_dir, input_vocab_size, output_vocab_size, padding_index, conf):
    """
    初始化或还原 transformer 。
    :param file: (str, mandatory) transformer 配置文件
    :param ckpt_dir: (str, mandatory) 检查点保存目录
    :param input_vocab_size: (int, mandatory) 输入词汇大小
    :param output_vocab_size: (int, mandatory) 输出词汇大小
    :param padding_index: (int, mandatory) 填充字符索引
    :param conf: (object) 配置信息
    :return: (LanguageTransformer) transformer 对象
    """
    if os.path.isfile(file):
        transformer = restore_transformer(file, ckpt_dir)
        if restore_transformer(file, ckpt_dir) is False:
            transformer = init_transformer(input_vocab_size, output_vocab_size, padding_index, conf)
    else:
        transformer = init_transformer(input_vocab_size, output_vocab_size, padding_index, conf)

    return transformer


class Language(object):

    def __init__(self, config=CONFIG_PATH, operation=M_PREDICTION):
        self.config = config
        self.operation = operation

        self.__init_config__()
        self.__init_dataset__()
        self.__init_or_restore_dict()
        self.__init_or_restore_model()

    def __init_config__(self):
        """
        初始化配置信息。将json配置转换为对象调用
        :return:
        """
        self.conf = Config(self.config).get_data()

    def __init_dataset__(self):
        """
        初始化数据集。如果 mode=train 则加载训练数据集。如果 mode=test 则加载测试数据集
        :return:
        """
        self.train_dataset = None
        self.test_dataset = None

        if self.operation == M_TRAIN:
            train_chinese_data, train_pinyin_data = load_dataset(self.conf.DATA_PATH_SOURCE, self.conf.DATA_PATH_TRAIN)
            valid_chinese_data, valid_pinyin_data = load_dataset(self.conf.DATA_PATH_SOURCE, self.conf.DATA_PATH_VALID)
            self.train_dataset = (train_chinese_data + valid_chinese_data, train_pinyin_data + valid_pinyin_data)
        elif self.operation == M_TEST:
            self.test_dataset = load_dataset(self.conf.DATA_PATH_SOURCE, self.conf.DATA_PATH_TEST)

    def __init_or_restore_dict(self):
        if self.operation == M_TRAIN:
            chinese_data, pinyin_data = self.train_dataset
        else:
            chinese_data, pinyin_data = None, None
        self.chinese_dict = init_or_restore_chinese_dict(chinese_data, self.conf.PATH_CHINESE_DICT)
        self.pinyin_dict = init_or_restore_pinyin_dict(pinyin_data, self.conf.PATH_PINYIN_DICT)

    def __init_or_restore_model(self):
        self.transformer = init_or_restore_transformer(self.conf.MODEL.TRANSFORMER.PATH_TRANSFORMER_CONFIG,
                                                       self.conf.MODEL.TRANSFORMER.ckpt_dir,
                                                       len(self.pinyin_dict),
                                                       len(self.chinese_dict),
                                                       self.pinyin_dict.PAD,
                                                       self.conf)

    def train(self):
        """训练"""
        chinese_data, pinyin_data = self.train_dataset
        x = [self.pinyin_dict.encoding(sent, alignment=True) for sent in tqdm(pinyin_data)]
        y = [self.chinese_dict.encoding(sent, alignment=True) for sent in tqdm(chinese_data)]
        self.transformer.fit(x=x, y=y,
                             batch_size=self.conf.MODEL.TRANSFORMER.BATCH_SIZE,
                             epochs=self.conf.MODEL.TRANSFORMER.EPOCHS,
                             validation_split=self.conf.MODEL.TRANSFORMER.VALIDATION_SPLIT,
                             shuffle=self.conf.MODEL.TRANSFORMER.SHUFFLE)
        self.transformer.dump_config(self.conf.MODEL.TRANSFORMER.PATH_TRANSFORMER_CONFIG)
        return True

    def test(self):
        """测试"""
        chinese_data, pinyin_data = self.test_dataset
        x = [self.pinyin_dict.encoding(sent, alignment=True) for sent in tqdm(pinyin_data)]
        y = [self.chinese_dict.encoding(sent, alignment=True) for sent in tqdm(chinese_data)]
        self.transformer.eval(x, y)
        return True

    def prediction(self, sent):
        """预测"""
        if not check_input_sentence(sent):
            return False
        sent = regular(sent)
        sent = self.pinyin_dict.encoding(sent)
        output, _ = self.transformer.predict(sent, y_start=self.chinese_dict.START, y_end=self.chinese_dict.END)
        output = self.chinese_dict.decoding(np.array(output))
        return output

    def run(self, sent=None):
        if self.operation == M_TRAIN:
            return self.train()
        elif self.operation == M_TEST:
            return self.test()
        elif self.operation == M_PREDICTION:
            return self.prediction(sent)


def test_module_func(operation):
    if operation == load_dataset.__name__:
        source_data = '../../data/data'
        train_data = '../../data/train'

        chinese_data, pinyin_data = load_dataset(source_data, train_data)
        for c_data, p_data in zip(chinese_data, pinyin_data):
            print('chinese data: ', c_data)
            print('pinyin data: ', p_data)
    if operation == 'train':
        language = Language(CONFIG_PATH, M_TRAIN)
        language.run()
    if operation == 'test':
        language = Language(CONFIG_PATH, M_TEST)
        language.run()
    if operation == 'pred':
        sent = 'lv4 shi4 yang2 chun1 yan1 jing3 da4 kuai4 wen2 zhang1 de5 di3 se4 si4 yue4 de5 lin2 '
        language = Language(CONFIG_PATH, M_PREDICTION)
        output = language.run(sent)
        print('sentence: ', sent)
        print('prediction: ', output)


def main():
    args = input_args()

    if args.mode == M_PREDICTION and check_input_sentence(args.sentence) is False:
        return False

    language = Language(CONFIG_PATH, args.mode)
    if args.mode == M_PREDICTION:
        output = language.run(args.sentence)
        print("sentence: ", args.sentence)
        print("prediction: ", output)
    else:
        language.run()


if __name__ == '__main__':
    main()
    # test_module_func(load_dataset.__name__)
    # test_module_func('train')
    # test_module_func('test')
    # test_module_func('pred')
