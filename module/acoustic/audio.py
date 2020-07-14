# -*- encoding: utf-8 -*-
"""
@file: audio.py
@time: 2020/5/21 下午4:16
@author: shenpinggang
@contact: 1285456152@qq.com
@desc: 提取音频特征。
"""
import os
import pickle
import numpy as np
import soundfile as sf
import scipy.io.wavfile as wav
from matplotlib import pyplot as plt
from python_speech_features import mfcc
from tqdm import tqdm

from module.core.utils import Writer
from module.core.utils import Features
from module.core.exception import exception_handling, FileNotFoundException, UnknownError, ParameterError


class AudioFeatures(Features):
    SPECTROGRAM = "spectrogram"
    MFCC = "mfcc"

    feature_type = [SPECTROGRAM, MFCC]

    def __init__(self, f_type=SPECTROGRAM, frame_length=256, frame_shift=128, mfcc_dim=13):
        super(AudioFeatures, self).__init__()
        # 帧长度。每个时序包含多个数据帧
        self.frame_length = frame_length

        # 帧移。每个时序移动多少数据帧。
        self.frame_shift = frame_shift

        # spectrogram 特征维度
        self.spectrogram_dim = self.frame_length // 2

        assert f_type in AudioFeatures.feature_type, \
            ParameterError("{} not in {}".format(f_type, AudioFeatures.feature_type))
        self.f_type = f_type

        # mfcc 特征维度
        self.mfcc_dim = mfcc_dim

        if self.f_type == AudioFeatures.SPECTROGRAM:
            self.mean = np.zeros(self.spectrogram_dim)
            self.std = np.ones(self.spectrogram_dim)
        elif self.f_type == AudioFeatures.MFCC:
            self.mean = np.zeros(self.mfcc_dim)
            self.std = np.ones(self.mfcc_dim)

        # 保存拟合语料中最大的序列长度
        self.max_length = 0

    def fit(self, train_data):
        """
        拟合数据。根据提供的数据初始化 mean 和 std
        :param train_data: (str of list, mandatory) 列表。元素是音频文件。['../data/A2_0.wav',...]
        :return:
        """
        train_feat = None
        if self.f_type == AudioFeatures.SPECTROGRAM:
            train_feat = [self.spectrogram(audio_file) for audio_file in tqdm(train_data)]
        elif self.f_type == AudioFeatures.MFCC:
            train_feat = [self.mfcc_feature(audio_file) for audio_file in tqdm(train_data)]

        for feat in train_feat:
            if feat.shape[0] > self.max_length:
                self.max_length = feat.shape[0]

        value = np.vstack(train_feat)
        self.mean = np.mean(value, axis=0)
        self.std = np.std(value, axis=0)

        print("audio features fit over! max length: {}".format(self.max_length))

    @exception_handling
    def spectrogram(self, file):
        """
        获取声谱特征
        :param file: (str, mandatory) 音频文件
        :return: (array) 声谱特征。shape:[时序, frame_length // 2]
        """
        assert os.path.isfile(file), FileNotFoundException(file)
        # 1.提取信号。
        # audio 为提取的数据。格式为[数据帧, 通道]。如果通道为0，则返回为1维度数据
        # sample_rate 为采样率。
        audio, sample_rate = sf.read(file, dtype='float32')

        assert audio.ndim == 1, UnknownError("check input data")

        # 2.分帧
        # 2.1 根据帧长度、帧移、信号数据。判断最多可以获得多少帧数据。对多余数据抛弃。
        # 例如信号数据 audio = [1,2,3,4,5,6,7,8,9]. 帧长度 = 4, 帧移 = 2
        # 则第一时序数据帧是 [1,2,3,4], 第二时序数据帧是：[3,4,5,6], 第三时序数据帧是：[5,6,7,8]. 其中 9 这个数据点被抛弃。总共 3 个时序
        trunc = (audio.shape[0] - self.frame_length) % self.frame_shift
        audio = audio[:audio.shape[0] - trunc]

        # 时序
        sequential = (audio.shape[0] - self.frame_length) // self.frame_shift
        assert sequential > 0, UnknownError("spectrogram feature sequence less than 0， "
                                            "audio file:{}, audio shape:{}".format(file, audio.shape))

        # 汉宁窗。shape: [frame_length]
        window = np.hanning(self.frame_length)

        # 3. 分帧 + 加窗 + 傅立叶变换
        # 3.1 定义保存数据
        feature = np.zeros((sequential, self.spectrogram_dim))
        for i in range(sequential):
            # 提取一个时序的帧数据
            seq_frame = audio[i * self.frame_shift: i * self.frame_shift + self.frame_length]
            # 加窗
            seq_frame = seq_frame * window
            # 傅立叶变换.
            seq_frame = np.fft.fft(seq_frame)
            # 然后取数据的一半。因为数据是对称的
            seq_frame = seq_frame[:self.spectrogram_dim]
            # 取绝对值
            feature[i] = np.abs(seq_frame) ** 2

        feature = feature + 1.e-12
        # assert feature.all() != 0, UnknownError("spectrogram feature cannot equal to 0, "
        #                                        "audio file:{}, audio shape:{}".format(file, audio.shape))

        feature = np.log(feature)
        return feature

    def mfcc_feature(self, file):
        """
        mfcc 特征
        :param file: (str, mandatory) 音频文件
        :return: (array) mfcc 特征
        """
        assert os.path.isfile(file), FileNotFoundException(file)

        (sample, audio) = wav.read(file)
        return mfcc(audio, sample, numcep=self.mfcc_dim)

    def normalized(self, feature, eps=1e-14):
        """
        特征归一化
        :param feature: (array, mandatory) 频谱特征
        :param eps: (float, optional, default=1e-14) 极小值
        :return: (array) 归一化特征
        """
        return (feature - self.mean) / (self.std + eps)

    def feature_alignment(self, file, alignment_size=None):
        """
        特征对齐。对特征序列大于 alignment_size 的进行截断。小于则使用 0 填充
        :param file: (str, mandatory) 音频文件
        :param alignment_size: (int, optional, default=None) 对齐大小。如果为None, 则使用拟合语料中得到的 self.max_length
        :return: (array) 填充序列和序列大小
        """
        alignment_size = self.max_length if alignment_size is None else alignment_size

        feature = np.zeros((1, 1))
        if self.f_type == AudioFeatures.SPECTROGRAM:
            feature = self.normalized(self.spectrogram(file))
        elif self.f_type == AudioFeatures.MFCC:
            feature = self.normalized(self.mfcc_feature(file))

        # 如果特征的序列大于 alignment_size。 则进行截断
        if feature.shape[0] > alignment_size:
            feature = feature[:alignment_size, :]

        padding_feature = np.zeros((alignment_size, self.feature_dim()))
        padding_feature[:feature.shape[0], :] = feature

        feature_length = feature.shape[0]
        return padding_feature, feature_length

    def feature(self, file, alignment=False, alignment_size=None):
        """
        获取特征。
        :param file: (str, mandatory) 音频文件
        :param alignment: (bool, optional, default=False) 是否对齐
        :param alignment_size: (int, optional, default=None) 对齐大小
        :return: (array) 特征数组
        """
        if alignment:
            feat, length = self.feature_alignment(file, alignment_size)
        else:
            feat = np.zeros((1, 1))
            length = 0
            if self.f_type == AudioFeatures.SPECTROGRAM:
                feat = self.normalized(self.spectrogram(file))
                length = feat.shape[0]

            elif self.f_type == AudioFeatures.MFCC:
                feat = self.normalized(self.mfcc_feature(file))
                length = feat.shape[0]

        return feat, length

    @staticmethod
    def plot_spectrogram(spectrogram_feature):
        """
        绘制声谱图
        :param spectrogram_feature: (array, mandatory) 声谱特征
        :return:
        """
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        ax.imshow(spectrogram_feature, aspect='auto')
        plt.title("Normalized spectrogram feature ")
        plt.ylabel("Time")
        plt.xlabel("Frequency")
        plt.show()

    @staticmethod
    def plot_mfcc(mfcc_feature):
        """
        绘制声谱图
        :param mfcc_feature: (array, mandatory) mfcc特征
        :return:
        """
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        ax.imshow(mfcc_feature, aspect='auto')
        plt.title("Normalized mfcc feature ")
        plt.ylabel("Time")
        plt.xlabel("Frequency")
        plt.show()

    @staticmethod
    def plot_audio(audio):
        """
        绘制信号图
        :param audio: (str) 音频信号文件
        :return:
        """
        audio_sig, _ = sf.read(audio, dtype='float32')
        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(111)
        steps = len(audio_sig)
        ax.plot(np.linspace(1, steps, steps), audio_sig)
        plt.title('Audio Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()

    def feature_dim(self):
        """
        特征维度
        :return: (int) 音频特征维度
        """
        if self.f_type == AudioFeatures.SPECTROGRAM:
            return self.spectrogram_dim
        elif self.f_type == AudioFeatures.MFCC:
            return self.mfcc_dim

    def save(self, file='audio_feature.pickle'):
        """
        保存字典
        :param file: (str, mandatory) 保存文件名
        :return:
        """
        file = file.strip()

        Writer.check_path(file)

        pickle.dump(self, open(file, 'wb'))

        print("save audio feature success! File: ", file)

    @staticmethod
    @exception_handling
    def load(file):
        """
        加载字典
        :param file: (str, mandatory) 字典文件
        :return: (dictionary) 字典对象
        """
        assert os.path.isfile(file), FileNotFoundException(file)

        with open(file, 'rb') as f_read:
            audio_feature = pickle.loads(f_read.read())

        return audio_feature


def test_module_func(operation):
    import os
    file_path = "../../data/data/"
    file_list = []
    for file_name in os.listdir(file_path):
        if file_name[-3:] == 'wav':
            file_list.append(os.path.join(file_path, file_name))

    if operation == 'spectrogram':
        feature_extraction = AudioFeatures(f_type=AudioFeatures.SPECTROGRAM)
        feature_extraction.fit(file_list)

        feat, source_length = feature_extraction.feature(file_list[0], alignment=False)
        assert feat.shape[0] == source_length

        feat, length = feature_extraction.feature(file_list[0], alignment=True)
        assert feat.shape[0] == feature_extraction.max_length
        assert feat.shape[1] == feature_extraction.feature_dim()
        assert length == source_length

        feat, length = feature_extraction.feature(file_list[0], alignment=True, alignment_size=1000)
        assert feat.shape[0] == 1000
        assert feat.shape[1] == feature_extraction.feature_dim()
        if source_length > 1000:
            assert length == 1000
        else:
            assert length == source_length

    elif operation == 'mfcc':
        feature_extraction = AudioFeatures(f_type=AudioFeatures.MFCC, mfcc_dim=13)
        feature_extraction.fit(file_list)

        feat, source_length = feature_extraction.feature(file_list[0], alignment=False)
        assert feat.shape[0] == source_length

        feat, length = feature_extraction.feature(file_list[0], alignment=True)
        assert feat.shape[0] == feature_extraction.max_length
        assert feat.shape[1] == feature_extraction.feature_dim()
        assert length == source_length

        feat, length = feature_extraction.feature(file_list[0], alignment=True, alignment_size=1000)
        assert feat.shape[0] == 1000
        assert feat.shape[1] == feature_extraction.feature_dim()
        if source_length > 1000:
            assert length == 1000
        else:
            assert length == source_length

    elif operation == 'plot_spectrogram':
        feature_extraction = AudioFeatures(f_type=AudioFeatures.SPECTROGRAM)
        feature_extraction.fit(file_list)

        feat, _ = feature_extraction.feature(file_list[0], alignment=False)
        AudioFeatures.plot_spectrogram(feat)

    elif operation == 'plot_mfcc':
        feature_extraction = AudioFeatures(f_type=AudioFeatures.MFCC, mfcc_dim=13)
        feature_extraction.fit(file_list)

        feat, _ = feature_extraction.feature(file_list[0], alignment=False)
        AudioFeatures.plot_mfcc(feat)


def main():
    # test_module_func('spectrogram')
    # test_module_func('mfcc')
    # test_module_func('plot_spectrogram')
    test_module_func('plot_mfcc')


if __name__ == "__main__":
    main()
