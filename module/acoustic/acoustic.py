# -*- encoding: utf-8 -*-
"""
@file: acoustic.py
@time: 2020/7/7 下午4:24
@author: shenpinggang
@contact: 1285456152@qq.com
@desc:
# 训练：python acoustic.py train
# 测试：python acoustic.py test
# 预测：python acoustic.py prediction --audio [file] --predict_mode [beam, greedy] --beam [int] --top [int]
    --audio:        指定预测的音频文件
    --predict_mode: 指定预测的模式。beam: 集束检索、 greedy: 贪婪检索
    --beam:         如果是集束检索，可指定检索束框。
    --top:          如果是集束检索，可指定返回的解码数量。

example:
    python acoustic.py prediction --audio ../../data/data/A2_0.wav --predict_mode greedy --beam 3 --top 2
"""
import re
import os
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm

try:
    from module.acoustic.audio import AudioFeatures
    from module.core.exception import FileNotFoundException
    from module.core.utils import DataUtils
    from module.core.exception import ParameterError
    from module.core.color import Color
    from module.core.utils import Config
    from module.acoustic.pinyin import PinYin
    from module.acoustic.transverse import AcousticTransverseNet
    from module.acoustic.transverse import virtual_data_generator
    from module.acoustic.transverse import virtual_dataset

except ModuleNotFoundError:
    import sys

    sys.path.append('../../')
    from module.acoustic.audio import AudioFeatures
    from module.core.exception import FileNotFoundException
    from module.core.utils import DataUtils
    from module.core.exception import ParameterError
    from module.core.color import Color
    from module.core.utils import Config
    from module.acoustic.pinyin import PinYin
    from module.acoustic.transverse import AcousticTransverseNet
    from module.acoustic.transverse import virtual_data_generator
    from module.acoustic.transverse import virtual_dataset

CONFIG_PATH = './config.json'

M_TRAIN = 'train'
M_TEST = 'test'
M_PREDICTION = 'prediction'

MODE = (M_TRAIN, M_TEST, M_PREDICTION)

M_PREDICT_BEAM = 'beam'
M_PREDICT_GREEDY = 'greedy'
M_PREDICT_MODE = [M_PREDICT_BEAM, M_PREDICT_GREEDY]


def input_args():
    parser = argparse.ArgumentParser(description=" 1.train  2.test  3.prediction ")
    parser.add_argument("mode", metavar='operation mode', choices=MODE, help="Select mode:{}".format(MODE))
    parser.add_argument("--audio", dest='audio', action='store', type=str, default='',
                        help="Please input prediction audio file !")
    parser.add_argument("--predict_mode", dest='predict_mode', choices=M_PREDICT_MODE,
                        help='Select predict mode: {}, default: greedy'.format(M_PREDICT_MODE),
                        default=M_PREDICT_GREEDY)
    parser.add_argument("--beam", dest='beam', action='store', type=int, default=3, help="Beam width, default: 3")
    parser.add_argument("--top", dest='top', action='store', type=int,
                        default=1, help='The number of cluster search results that need to be returned. default=1')
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
            # assert len(chinese) == len(pinyin), "The number of pinyin:{} and chinese:{} is different, " \
            #                                     "chinese:{}, pinyin:{}, file:{}".format(len(pinyin), len(chinese),
            #                                                                             chinese, pinyin,
            #                                                                             label_data_file_path)

            audio_data.append(audio_file_path)
            pinyin_data.append(pinyin)
            chinese_data.append(chinese)

    return audio_data, pinyin_data, chinese_data


def init_or_restore_pinyin_dict(dataset, file):
    """
    初始化或还原拼音字典
    :param dataset:  (list, mandatory) 数据集
    :param file: (str, mandatory) 保存或者还原pinyin字典的文件
    :return: (WordDict) 字典对象
    """
    if os.path.isfile(file):
        pinyin = PinYin.load(file)
        output = 'Restore pinyin dict from file:{}'.format(file)
    else:
        pinyin = PinYin()
        pinyin.fit(dataset=dataset)
        pinyin.save(file)
        output = 'Initialize pinyin dict form dataset'

    print(Color.green(output))
    return pinyin


def init_or_restore_audio_feature(dataset, file, f_type, frame_length, frame_shift, mfcc_dim):
    """
    初始化或还原音频特征器
    :param dataset: (list, mandatory) 数据集
    :param file: (str, mandatory) 保存或者还原audio_feature的文件
    :param f_type: (str, mandatory) 处理音频特征的类型。"spectrogram" or "mfcc"
    :param frame_length: (int, mandatory) 帧长
    :param frame_shift: (int, mandatory) 帧移
    :param mfcc_dim: (int, mandatory) mfcc 特征维度
    :return: (AudioFeature) 音频特征对象
    """
    if os.path.isfile(file):
        audio_feat = AudioFeatures.load(file)
        output = "Audio feature: restore from : {}".format(file)
    else:
        audio_feat = AudioFeatures(f_type=f_type,
                                   frame_length=frame_length,
                                   frame_shift=frame_shift,
                                   mfcc_dim=mfcc_dim)
        audio_feat.fit(train_data=dataset)
        output = "Audio feature: Not found file:{} Initializing from scratch".format(file)
        audio_feat.save(file)

    print(Color.green(output))
    return audio_feat


def init_transverse(conf, input_dim, output_vocab_size):
    """
    初始化 Transverse
    :param conf: (obj, mandatory) 配置对象
    :param input_dim: (int, mandatory) 输入维度
    :param output_vocab_size: (int, mandatory) 输出词汇量大小
    :return: (AcousticTransverseNet) AcousticTransverseNet 对象
    """
    transverse = AcousticTransverseNet(input_dim=input_dim,
                                       output_vocab_size=output_vocab_size,
                                       dn_hidden_size=conf.MODEL.TRANSVERSE.dn_hidden_size,
                                       dilated_conv_depth=conf.MODEL.TRANSVERSE.dilated_conv_depth,
                                       width_conv_depth=conf.MODEL.TRANSVERSE.width_conv_depth,
                                       multi_dilated_rate=conf.MODEL.TRANSVERSE.multi_dilated_rate,
                                       dilated_conv_filters=conf.MODEL.TRANSVERSE.dilated_conv_filters,
                                       width_conv_filters=conf.MODEL.TRANSVERSE.width_conv_filters,
                                       dropout_rate=conf.MODEL.TRANSVERSE.dropout_rate,
                                       l1=conf.MODEL.TRANSVERSE.l1,
                                       l2=conf.MODEL.TRANSVERSE.l2,
                                       activation=conf.MODEL.TRANSVERSE.activation,
                                       learning_rate=conf.MODEL.TRANSVERSE.learning_rate,
                                       warmup_steps=conf.MODEL.TRANSVERSE.warmup_steps,
                                       optimizer_beta_1=conf.MODEL.TRANSVERSE.optimizer_beta_1,
                                       optimizer_beta_2=conf.MODEL.TRANSVERSE.optimizer_beta_2,
                                       optimizer_epsilon=conf.MODEL.TRANSVERSE.optimizer_epsilon,
                                       ckpt_dir=conf.MODEL.TRANSVERSE.ckpt_dir,
                                       ckpt_max_to_keep=conf.MODEL.TRANSVERSE.ckpt_max_to_keep)
    print(Color.green('Initialization transverse from scratch'))
    return transverse


def restore_transverse(config, ckpt_dir):
    """
    还原 transverse
    :param config: (str, mandatory) 配置文件
    :param ckpt_dir: (str, mandatory) 检查点保存目录
    :return: (AcousticTransverseNet) AcousticTransverseNet 对象
    """
    try:
        transverse = AcousticTransverseNet.restore(config, ckpt_dir)
    except Exception as e:
        print("restore AcousticTransverseNet fail! config:{}, checkpoint:{}".format(config, ckpt_dir))
        print(e)
        return False

    return transverse


def init_or_restore_transverse(file, ckpt_dir, conf, input_dim, output_vocab_size):
    """
    初始化或还原 transverse 对象。如果存在配置文件和检查点，则还原模型。否则初始化
    :param file: (str, mandatory) 配置文件
    :param ckpt_dir: (str, mandatory) 检查点保存目录
    :param conf: (obj, mandatory) 配置对象
    :param input_dim: (int, mandatory) 输入维度
    :param output_vocab_size: (int, mandatory) 输出词汇量大小
    :return: (AcousticTransverseNet) AcousticTransverseNet 对象
    """
    if os.path.isfile(file):
        transverse = restore_transverse(file, ckpt_dir)
        if not transverse:
            transverse = init_transverse(conf, input_dim, output_vocab_size)
    else:
        transverse = init_transverse(conf, input_dim, output_vocab_size)

    return transverse


def packaging_dataset(audio_data, pinyin_data, audio_feature, pinyin_dict, batch_size=2, shuffle=True):
    """
    包装数据集
    :param audio_data: (list, mandatory) 音频数据
    :param pinyin_data: (list, mandatory) 拼音标签数据
    :param audio_feature: (AudioFeature, mandatory) 音频数据特征提取
    :param pinyin_dict: (PinYin, mandatory) 拼音数据提取
    :param batch_size: (int, optional, default=2) 批量样本
    :param shuffle: (bool, optional, default=True) 是否洗牌
    :return: (tf.data.Dataset) 数据集
    """
    x_data, input_data, y_data, label_data = list(), list(), list(), list()
    for a_data, p_data in tqdm(zip(audio_data, pinyin_data)):
        x, input_length = audio_feature.feature(a_data, alignment=True)
        y, label_length = pinyin_dict.encoding(p_data, alignment=True)

        input_length = [input_length]
        label_length = [label_length]

        x_data.append(x.astype(np.float32).tolist())
        input_data.append(input_length)
        y_data.append(y.astype(np.int32).tolist())
        label_data.append(label_length)

    x = tf.convert_to_tensor(x_data, dtype=tf.float32)
    input_length = tf.convert_to_tensor(input_data, dtype=tf.int32)
    y = tf.convert_to_tensor(y_data, dtype=tf.int32)
    label_length = tf.convert_to_tensor(label_data, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((x, input_length, y, label_length))
    if shuffle:
        dataset = dataset.shuffle(batch_size * 2).batch(batch_size)
    else:
        dataset = dataset.batch(batch_size)

    return dataset


class DataGenerator(object):

    def __init__(self, audio_data, pinyin_data, pinyin_dict, audio_feature):
        """
        数据生成器
        :param audio_data: (list, mandatory) 音频数据
        :param pinyin_data: (list, mandatory) 拼音数据
        :param pinyin_dict: (PinYin, mandatory) 拼音字典对象
        :param audio_feature: (AudioFeature, mandatory) 音频特征处理对象
        :return: (array, list, array, list) 单个样本音频特征、音频长度、拼音数据、拼音长度
        """
        self.audio_data = audio_data
        self.pinyin_data = pinyin_data
        self.pinyin_dict = pinyin_dict
        self.audio_feature = audio_feature

    def next(self):
        for a_data, p_data in zip(self.audio_data, self.pinyin_data):
            x, input_length = self.audio_feature.feature(a_data, alignment=True)
            y, label_length = self.pinyin_dict.encoding(p_data, alignment=True)

            input_length = [input_length]
            label_length = [label_length]

            yield x, input_length, y, label_length


class Acoustic(object):
    def __init__(self, config=CONFIG_PATH, operation=M_PREDICTION):
        self.config = config
        self.operation = operation

        self.__init_config__()
        self.__load_dataset__()
        self.__init_audio_and_pinyin__()
        self.__init_transverse__()
        self.__init_dataset__()

    def __init_config__(self):
        self.conf = Config(self.config).get_data()

    def __load_dataset__(self):
        self.train_dataset, self.test_dataset = None, None
        if self.operation == M_TRAIN:
            train_audio_data, train_pinyin_data, _ = load_dataset(self.conf.DATA_PATH_SOURCE, self.conf.DATA_PATH_TRAIN)
            valid_audio_data, valid_pinyin_data, _ = load_dataset(self.conf.DATA_PATH_SOURCE, self.conf.DATA_PATH_VALID)
            self.train_dataset = (train_audio_data + valid_audio_data, train_pinyin_data + valid_pinyin_data)
        elif self.operation == M_TEST:
            test_audio_data, test_pinyin_data, _ = load_dataset(self.conf.DATA_PATH_SOURCE, self.conf.DATA_PATH_TEST)
            self.test_dataset = (test_audio_data, test_pinyin_data)

    def __init_audio_and_pinyin__(self):
        if self.operation == M_TRAIN:
            audio_data, pinyin_data = self.train_dataset
        else:
            audio_data, pinyin_data = None, None
        self.audio_feature = init_or_restore_audio_feature(audio_data,
                                                           self.conf.AUDIO.PATH_AUDIO_FEATURE,
                                                           self.conf.AUDIO.f_type,
                                                           self.conf.AUDIO.frame_length,
                                                           self.conf.AUDIO.frame_shift,
                                                           self.conf.AUDIO.mfcc_dim)
        self.pinyin_dict = init_or_restore_pinyin_dict(pinyin_data, self.conf.PATH_PINYIN_DICT)

    def __init_transverse__(self):
        self.model = init_or_restore_transverse(self.conf.MODEL.TRANSVERSE.PATH_TRANSVERSE_CONFIG,
                                                self.conf.MODEL.TRANSVERSE.ckpt_dir,
                                                self.conf,
                                                self.audio_feature.feature_dim(),
                                                len(self.pinyin_dict) + 1)

    def __init_dataset__(self):
        self.dataset = None
        if self.operation == M_TRAIN:
            train_audio_data, train_pinyin_data = self.train_dataset
            self.dataset = packaging_dataset(train_audio_data, train_pinyin_data, self.audio_feature, self.pinyin_dict,
                                             shuffle=self.conf.MODEL.TRANSVERSE.SHUFFLE,
                                             batch_size=self.conf.MODEL.TRANSVERSE.BATCH_SIZE)

        elif self.operation == M_TEST:
            test_audio_data, test_pinyin_data = self.test_dataset
            self.dataset = packaging_dataset(test_audio_data, test_pinyin_data, self.audio_feature,
                                             self.pinyin_dict, shuffle=False, batch_size=1)

    def train(self):
        """ 训练 """
        train_loss, valid_loss = self.model.fit(self.dataset,
                                                epochs=self.conf.MODEL.TRANSVERSE.EPOCHS,
                                                validation_split=self.conf.MODEL.TRANSVERSE.VALIDATION_SPLIT)
        self.model.dump_config(self.conf.MODEL.TRANSVERSE.PATH_TRANSVERSE_CONFIG)
        return train_loss, valid_loss

    def test(self):
        """ 测试 """
        test_loss = self.model.eval(self.dataset)
        return test_loss

    def predict(self, audio, beam_width=3, top_paths=1, greedy=True):
        """ 预测 """
        assert os.path.isfile(audio), FileNotFoundException(audio)
        features, input_length = self.audio_feature.feature(audio, alignment=False)

        features = features[np.newaxis, :, :]
        features = tf.convert_to_tensor(features, dtype=tf.float32)
        input_length = tf.reshape(tf.convert_to_tensor([input_length], dtype=tf.int32), shape=[1, 1])

        sequence, _ = self.model.predict(features, input_length, beam_width=beam_width, top_paths=top_paths,
                                         greedy=greedy)
        if greedy:
            output = self.pinyin_dict.decoding(sequence)
        else:
            output = [self.pinyin_dict.decoding(seq) for seq in sequence]

        return output

    def run(self, audio=None, beam_width=3, top_paths=1, greedy=True):
        if self.operation == M_TRAIN:
            return self.train()
        elif self.operation == M_TEST:
            return self.test()
        elif self.operation == M_PREDICTION:
            return self.predict(audio, beam_width, top_paths, greedy)
        else:
            return None


def test_module_func(operation):
    if operation == load_dataset.__name__:
        source_data = '../../data/data'
        train_data = '../../data/train'
        audio_data, pinyin_data, chinese_data = load_dataset(source_data, train_data)
        for a_data, p_data, c_data in zip(audio_data, pinyin_data, chinese_data):
            print('audio:{}, pinyin:{}, chinese:{}'.format(a_data, p_data, c_data))
            break

    elif operation == DataGenerator.__name__:
        source_data = '../../data/data'
        train_data = '../../data/train'
        audio_data, pinyin_data, chinese_data = load_dataset(source_data, train_data)
        pinyin_dict = init_or_restore_pinyin_dict(pinyin_data, file='./save/pinyin.pickle')
        audio_feature = init_or_restore_audio_feature(audio_data, file='./save/audio_feature.pickle', f_type='mfcc',
                                                      frame_length=256,
                                                      frame_shift=128, mfcc_dim=13)

        data_generator = DataGenerator(audio_data, pinyin_data, pinyin_dict, audio_feature)

        for x, input_length, y, label_length in data_generator.next():
            print("Generator x shape: ", x.shape)
            print("Generator y shape: ", y.shape)
            print("Generator input length: ", input_length)
            print("Generator label length: ", label_length)

    elif operation == init_or_restore_transverse.__name__:
        x_dim = 13
        y_range = 1000
        dataset = virtual_dataset(16, x_length=1024, y_length=32, x_dim=x_dim, y_range=y_range, batch_size=2,
                                  shuffle=True)

        conf = Config(CONFIG_PATH).get_data()
        model = init_or_restore_transverse(file='./test_model/conf.json', ckpt_dir='./test_model/ckpt',
                                           conf=conf,
                                           input_dim=x_dim,
                                           output_vocab_size=y_range + 1)
        model.fit(dataset, epochs=2)

    elif operation == 'fit_generator':
        x_dim = 13
        y_range = 1000

        conf = Config(CONFIG_PATH).get_data()
        model = init_or_restore_transverse(file='./test_model/conf.json', ckpt_dir='./test_model/ckpt',
                                           conf=conf,
                                           input_dim=x_dim,
                                           output_vocab_size=y_range + 1)
        model.fit_generator(virtual_data_generator, batch_size=2, shuffle=True, epochs=2, args=[16, 1024, 32, 13, 1000])

    elif operation == packaging_dataset.__name__:
        # model = Acoustic(CONFIG_PATH, M_TRAIN)
        # model.run()
        conf = Config(CONFIG_PATH).get_data()
        audio_data, pinyin_data, _ = load_dataset(conf.DATA_PATH_SOURCE, conf.DATA_PATH_TRAIN)

        audio_feature = init_or_restore_audio_feature(audio_data,
                                                      conf.AUDIO.PATH_AUDIO_FEATURE,
                                                      conf.AUDIO.f_type,
                                                      conf.AUDIO.frame_length,
                                                      conf.AUDIO.frame_shift,
                                                      conf.AUDIO.mfcc_dim)
        pinyin_dict = init_or_restore_pinyin_dict(pinyin_data, conf.PATH_PINYIN_DICT)

        dataset = packaging_dataset(audio_data, pinyin_data, audio_feature=audio_feature, pinyin_dict=pinyin_dict,
                                    shuffle=True, batch_size=2)
        for x, input_length, y, label_length in dataset:
            print("x shape: ", x.shape)
            print("input length: ", input_length)
            print("y: ", y)
            print("label length: ", label_length)

        # model = init_or_restore_transverse(file='./test_model/conf.json', ckpt_dir='./test_model/ckpt',
        #                                    conf=conf,
        #                                    input_dim=audio_feature.feature_dim(),
        #                                    output_vocab_size=len(pinyin_dict) + 1)
        # model.fit(dataset, epochs=2)

    elif operation == 'train':
        model = Acoustic(CONFIG_PATH, M_TRAIN)
        model.run()

    elif operation == 'test':
        model = Acoustic(CONFIG_PATH, M_TEST)
        model.run()

    elif operation == 'prediction':
        audio = '../../data/data/A2_0.wav'
        model = Acoustic(CONFIG_PATH, M_PREDICTION)
        print("input: {}".format(audio))

        output = model.run(audio, greedy=True)
        print("Greedy predict: {}".format(output))

        top = 3
        output = model.run(audio, beam_width=3, top_paths=top, greedy=False)
        for i in range(top):
            print("Beam predict: {}.{}".format(i, output[i]))


def main():
    args = input_args()

    if args.mode == M_PREDICTION:
        if not check_input_args(args):
            return False

        audio = args.audio
        acoustic = Acoustic(CONFIG_PATH, operation=M_PREDICTION)
        if args.predict_mode == M_PREDICT_GREEDY:
            output = acoustic.run(audio=audio, greedy=True)
            print(Color.red("input: {}".format(audio)))
            print(Color.yellow("Greedy predict: {}".format(output)))
        elif args.predict_mode == M_PREDICT_BEAM:
            output = acoustic.run(audio=audio, beam_width=args.beam, top_paths=args.top, greedy=False)
            print(Color.yellow("input: {}".format(audio)))
            for i in range(args.top):
                print(Color.yellow("Beam predict: {}.{}".format(i, output[i])))
                print()
    else:
        acoustic = Acoustic(CONFIG_PATH, operation=args.mode)
        print(acoustic.model.summary())
        acoustic.run()


if __name__ == '__main__':
    # test_module_func(load_dataset.__name__)
    # test_module_func(DataGenerator.__name__)
    # test_module_func(init_or_restore_transverse.__name__)
    # test_module_func(packaging_dataset.__name__)
    # test_module_func('train')
    # test_module_func('test')
    # test_module_func('prediction')
    main()
