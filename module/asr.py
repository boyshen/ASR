# -*- encoding: utf-8 -*-
"""
@file: asr.py
@time: 2020/7/13 下午2:42
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 评估：python asr.py eval
    使用词错误率 WER 对测试数据集进行评估，输出词错误率
# 预测：python asr.py prediction --audio [file] --acoustic_mode [greedy, beam] --beam [int] --top [int]
    --audio         指定预测音频文件
    --acoustic      指定声学模型预测模式。greedy：贪婪检索、beam：集束检索
    --beam          集束检索指定束宽
    --top           返回集束检索的结果数量。top 的值需要小于 beam

example:
    贪婪检索： python asr.py prediction --audio ../data/data/A2_0.wav --acoustic_mode greedy
    集束检索： python asr.py prediction --audio ../data/data/A2_0.wav --acoustic_mode beam --beam 3 --top 2
"""
import sys
import os
import re
import argparse
from tqdm import tqdm

try:
    from collections import Counter
    from module.core.utils import Config
    from module.acoustic.acoustic import Acoustic
    from module.language.language import Language
    from module.core.utils import DataUtils
    from module.core.exception import FileNotFoundException
    from module.core.exception import UnknownError
    from module.core.color import Color
except ModuleNotFoundError:
    sys.path.append('../')
    from collections import Counter
    from module.core.utils import Config
    from module.acoustic.acoustic import Acoustic
    from module.language.language import Language
    from module.core.utils import DataUtils
    from module.core.exception import FileNotFoundException
    from module.core.exception import UnknownError
    from module.core.color import Color

M_EVAL = 'eval'
M_PREDICTION = 'prediction'
MODE = [M_EVAL, M_PREDICTION]

M_PREDICT_BEAM = 'beam'
M_PREDICT_GREEDY = 'greedy'
M_PREDICT_MODE = [M_PREDICT_BEAM, M_PREDICT_GREEDY]

CONFIG = './config.json'


def input_args():
    parser = argparse.ArgumentParser(description="1. eval  2. predict")
    parser.add_argument('mode', metavar='operation mode', choices=MODE, help="Select mode:{}".format(MODE))
    parser.add_argument('--audio', dest='audio', action='store', type=str, default='',
                        help="Please input prediction audio file")
    parser.add_argument('--acoustic_mode', dest='acoustic_mode', choices=M_PREDICT_MODE, default=M_PREDICT_GREEDY,
                        help='Select predict mode:{}, default: greedy')
    parser.add_argument('--beam', dest='beam', action='store', type=int, default=3, help="Beam width, default:3 ")
    parser.add_argument('--top', dest='top', action='store', type=int, default=3,
                        help="The number of cluster search results that need to be returned. default=1")

    args = parser.parse_args()
    return args


def check_input_args(args):
    """ 检查输入的参数 """
    if args.audio is '':
        print(Color.red("Please input audio file !"))
        return False
    else:
        if not os.path.isfile(args.audio):
            print(FileNotFoundException(args.audio))
            return False

    if args.top > args.beam:
        print(Color.red("requested top:{} than the beam:{}".format(args.top, args.beam)))
        return False

    return True


def handle(line):
    """
    处理每行文本的函数
    :param line: (str, mandatory) 字符／句子。例如 hello word
    :return: (list) 列表。例如 [hello, world]
    """
    line_data = list()
    for word in line.split(' '):
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
    audio_data, pinyin_data, chinese_data = list(), list(), list()

    data_file = os.listdir(data_path)
    for file in tqdm(data_file):
        # 检查文件名后缀是不是 wav 音频类型文件。
        if file[-3:] == 'wav':
            # 标签文件。根据音频文件名得到标签文件名
            label_file = file + '.trn'

            # 判断标签文件是否存在。不存在则报错
            assert label_file in data_file, FileNotFoundException(label_file)

            # 音频文件路径和标签文件路径
            audio_file_path = os.path.join(data_path, file)
            label_file_path = os.path.join(data_path, label_file)

            # 读取标签文件内容。找到对应标签数据路径
            label_data_file_path = DataUtils.read_text_data(label_file_path, show_progress_bar=False)
            assert len(label_data_file_path) == 1, \
                'Get redundant data path: {}, label_file_path:{}'.format(label_data_file_path, label_file_path)

            # 重新拼接路径
            label_data_file_path = os.path.join(source_data_path, label_data_file_path[0].split('/')[-1])
            assert os.path.isfile(label_data_file_path), FileNotFoundException(label_data_file_path)

            # 读取标签数据。包括字符、拼音
            text_data = DataUtils.read_text_data(label_data_file_path, handle_func=handle, show_progress_bar=False)
            chinese = text_data[0]
            pinyin = text_data[1]

            # 检查pinyin字符是否正确。主要判断每个拼音的最后一个字符是不是数字。不是数字则报错
            for py in pinyin:
                assert py[-1].isdigit(), "The last character:{} of Pinyin is not a number! " \
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
            assert len(chinese) == len(pinyin), "The number of pinyin:{} and chinese:{} is different, " \
                                                "chinese:{}, pinyin:{}, file:{}".format(len(pinyin), len(chinese),
                                                                                        chinese, pinyin,
                                                                                        label_data_file_path)

            audio_data.append(audio_file_path)
            pinyin_data.append(pinyin)
            chinese_data.append(chinese)

    return audio_data, pinyin_data, chinese_data


def word_error_rate(y_true, y_pred):
    """
    计算单词的错误率
    :param y_true: (list, mandatory) 正确的词汇
    :param y_pred: (list, mandatory) 预测的词汇
    :return:
    """
    insertions_word = 0
    substitutions_word = 0
    deletions_word = 0
    total_word = len(y_pred)

    # 获取删除的词数量
    if len(y_true) > len(y_pred):
        deletions_word = len(y_true) - len(y_pred)
        y_true = y_true[:len(y_pred)]

    # 获取插入多余的词数量
    elif len(y_true) < len(y_pred):
        insertions_word = len(y_pred) - len(y_true)
        y_pred = y_pred[:len(y_true)]

    assert len(y_true) == len(y_pred), \
        UnknownError("y_true:{} and y_pred:{} size must be equal".format(len(y_true), len(y_pred)))

    # 获取替换的词数量
    for true_word, pred_word in zip(y_true, y_pred):
        if true_word != pred_word:
            substitutions_word += 1

    return (insertions_word + substitutions_word + deletions_word) / total_word


class ChineseDict(object):
    """ 中文字典对象 用于将中文字符转换成数字token，方便进行比较。"""

    def __init__(self):
        self.chinese_dict = {}

        self.UNK = 0
        self.UNK_PAG = '<UNK>'

    def fit(self, dataset):
        counter = Counter()
        for data in tqdm(dataset):
            counter.update(data)

        for word, _ in counter.items():
            self.chinese_dict[word] = len(self.chinese_dict)
        self.chinese_dict[self.UNK_PAG] = len(self.chinese_dict)
        self.UNK = self.chinese_dict[self.UNK_PAG]

    def word_to_token(self, word):
        try:
            token = self.chinese_dict[word]
        except KeyError:
            token = self.UNK
        return token

    def encoding(self, sent):
        return [self.word_to_token(word) for word in sent]


class Asr(object):
    def __init__(self, config, operation):
        self.config = config
        self.operation = operation

        self.__init_config__()
        self.__init_dataset__()
        self.__init_acoustic_model__()
        self.__init_language_model__()

    def __init_config__(self):
        self.conf = Config(self.config).get_data()

    def __init_dataset__(self):
        self.dataset = None
        if self.operation == M_EVAL:
            self.dataset = load_dataset(self.conf.DATA_PATH_SOURCE, self.conf.DATA_PATH_TEST)

    def __init_acoustic_model__(self):
        self.acoustic = Acoustic(self.conf.CONFIG_ACOUSTIC, operation=M_PREDICTION)

    def __init_language_model__(self):
        self.language = Language(self.conf.CONFIG_LANGUAGE, operation=M_PREDICTION)

    def eval(self):
        """ 评估。计算输出音频的词平均错误率 """
        audio_data, pinyin_data, chinese_data = self.dataset
        wer = 0
        for step, (a_data, c_data) in enumerate(zip(audio_data, chinese_data)):
            try:
                predict_pinyin = self.acoustic.run(a_data, greedy=True)
                predict_chinese = self.language.run(' '.join(predict_pinyin))
            except Exception as e:
                print(e)
                print('audio data:', a_data)
                print('chinese data:', c_data)
                continue

            wer += word_error_rate(c_data, predict_chinese)
            avg_wer = wer / (step + 1)
            template = "Eval: {}/{}, Avg word error rate:{:.4f}".format(step + 1, len(audio_data), avg_wer)
            sys.stdout.write('\r' + template)
            sys.stdout.flush()
        print()

    def predict(self, audio, acoustic_mode=M_PREDICT_GREEDY, beam=3, top=1):
        """ 预测。 输入音频文件，返回中文字符 """
        predict_chinese = list()
        if acoustic_mode == M_PREDICT_GREEDY:
            predict_pinyin = self.acoustic.run(audio, beam_width=beam, top_paths=top, greedy=True)
            predict_chinese = self.language.run(' '.join(predict_pinyin))
        elif acoustic_mode == M_PREDICT_BEAM:
            predict_pinyin = self.acoustic.run(audio, beam_width=beam, top_paths=top, greedy=False)
            predict_chinese = [self.language.run(' '.join(pinyin)) for pinyin in predict_pinyin]

        return predict_chinese

    def run(self, audio=None, acoustic_mode=M_PREDICT_GREEDY, beam=3, top=1):
        if self.operation == M_EVAL:
            return self.eval()
        elif self.operation == M_PREDICTION:
            return self.predict(audio, acoustic_mode, beam, top)
        else:
            return None


def test_module_func(operation):
    if operation == load_dataset.__name__:
        source_data_path = '../data/data'
        test_data_path = '../data/test'

        audio_data, pinyin_data, chinese_data = load_dataset(source_data_path, test_data_path)
        for i, (a_data, p_data, c_data) in enumerate(zip(audio_data, pinyin_data, chinese_data)):
            if i == 3:
                break
            print("audio: ", a_data)
            print("pinyin: ", p_data)
            print("chinese: ", c_data)

    elif operation == word_error_rate.__name__:
        y_true = ['爱', '国', '将', '士']
        y_pred = ['爱', '国', '将', '士', '马', '占', '山']
        print("word error rate: {:.4f}".format(word_error_rate(y_true, y_pred)))

        y_true = ['也', '奋', '起', '抗', '战']
        y_pred = ['也', '奋', '起']
        print("word error rate: {:.4f}".format(word_error_rate(y_true, y_pred)))

        y_true = ['藏', '起', '来', '几', '次', '围', '捕']
        y_pred = ['王', '英', '汉', '被', '枪', '毙', '后']
        print("word error rate: {:.4f}".format(word_error_rate(y_true, y_pred)))

        y_true = ['奉', '献', '的', '人', '们']
        y_pred = ['奉', '献', '的', '我', '们']
        print("word error rate: {:.4f}".format(word_error_rate(y_true, y_pred)))

    elif operation == ChineseDict.__name__:
        data = ['绿', '是', '阳', '春', '烟', '景', '大', '块', '文', '章', '的', '底', '色',
                '四', '月', '的', '林', '峦', '更', '是', '绿', '得', '鲜', '活', '秀', '媚', '诗', '意', '盎', '然']
        chinese_dict = ChineseDict()
        chinese_dict.fit(data)

        print(chinese_dict.chinese_dict)
        assert chinese_dict.word_to_token(chinese_dict.UNK_PAG) == 27
        assert chinese_dict.encoding(['绿', '是', '阳', '春']) == [0, 1, 2, 3]

    elif operation == 'eval':
        asr = Asr(CONFIG, M_EVAL)
        asr.run()

    elif operation == 'predict':
        audio = '../data/data/A2_0.wav'
        asr = Asr(CONFIG, M_PREDICTION)
        output = asr.run(audio, M_PREDICT_GREEDY)
        print("input: {}".format(audio))
        print("Greedy predict: {}".format(output))

        output = asr.run(audio, M_PREDICT_BEAM, beam=3, top=1)
        print("Beam predict: {}".format(output))


def main():
    args = input_args()

    if args.mode == M_PREDICTION:
        if check_input_args(args) is False:
            return False

        asr = Asr(CONFIG, M_PREDICTION)
        output = asr.run(args.audio, args.acoustic_mode, args.beam, args.top)
        print("Input: {}".format(args.audio))
        if args.acoustic_mode == M_PREDICT_GREEDY:
            print("Predict: {}".format(output))
        elif args.acoustic_mode == M_PREDICT_BEAM:
            for i in range(args.top):
                print("Predict: {}.{}".format(i, output[i]))

    elif args.mode == M_EVAL:
        asr = Asr(CONFIG, M_EVAL)
        asr.run()


if __name__ == '__main__':
    # test_module_func(load_dataset.__name__)
    # test_module_func(ChineseDict.__name__)
    # test_module_func(word_error_rate.__name__)
    # test_module_func('eval')
    # test_module_func('predict')
    main()
