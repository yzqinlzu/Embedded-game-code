from pyAudioAnalysis import audioFeatureExtraction
import wave
import numpy as np
from sklearn.externals import joblib
from pyAudioAnalysis import audioBasicIO
import librosa
#获取要验证的音频的特征
win_size = 0.04
step = 0.01
i = 0
emclass = {0:'W', 1:'L', 2:'E', 3:'A', 4:'F', 5:'T', 6:'N'}
cn={'W':'愤怒','L':'无聊','E':'厌恶','A':'恐惧','F':'快乐','T':'悲伤','N':'中性'}
pp =joblib.load('cnpp.model')#读取预处理器
estimator = joblib.load('cnmodel.plk')#读取预测模型  #中文识别模型

def wav_predict(path):
    '''
    识别接口，给一个wav 返回情绪标签
    :param path:
    :return:
    '''
    #获取音频特征
    [Fs,x] = audioBasicIO.readAudioFile(path)
    print(Fs,x)
    x.reshape(1,-1)
    print(x)
    F = audioFeatureExtraction.stFeatureExtraction(x, Fs, win_size * Fs, step * Fs)
    Ftest = np.concatenate((np.mean(F[0], axis=1), np.std(F[0], axis=1))) #计算特征？？平均值标准差？？
    #特征处理
    Ftest=[Ftest]
    Ftest=np.array(Ftest)
    Ftest.reshape(1, -1)
    Ftest = pp.standardize_one(Ftest)
    Ftest = pp.project_on_pc_test(Ftest)
    #预测类别
    y_predict = estimator.predict(Ftest)
    return emclass[y_predict[0]]
