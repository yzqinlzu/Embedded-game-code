import wave
import numpy as np
import nextpow2
import math




def berouti(SNR):
    #     a = 0
    #     print(SNR)
    if -5.0 <= SNR <= 20.0:
        a = 4 - SNR * 3 / 20
    #         print("123")
    else:
        if SNR <= -5.0:
            a = 5
        #             print("456")
        if SNR >= 20:
            a = 1
    #             print("789")
    #     print("111")
    return a


def berouti1(SNR):
    #     a = 0
    if -5.0 <= SNR <= 20.0:
        a = 3 - SNR * 2 / 20
    else:
        if SNR <= -5.0:
            a = 4
        if SNR >= 20:
            a = 1
    return a


def find_index(x_list):
    index_list = []
    for i in range(len(x_list)):
        if x_list[i] < 0:
            index_list.append(i)
    return index_list



def myreduction(filename):
    f = wave.open(filename)  # 读取数据


    # 读取格式信息
    # (nchannels, sampwidth, framerate, nframes, comptype, compname)
    params = f.getparams()
    # params
    nchannels, sampwidth, framerate, nframes = params[:4]
    # nchannels, sampwidth, framerate, nframes


    # 读取波形数据
    str_data = f.readframes(nframes)[90000:]
    print(str_data)
    f.close()
    # str_data


    # 将波形数据转换为数组
    x = np.fromstring(str_data, dtype=np.short)
    # x


    # 计算参数
    len_ = 20 * framerate // 1000
    PERC = 50
    len1 = len_ * PERC // 100
    len2 = len_ - len1
    # 设置默认参数
    Thres = 3
    Expnt = 2.0
    beta = 0.002
    G = 0.9


    # 初始化汉明窗
    win = np.hamming(len_)
    # win


    # 重叠的标准化增益+50%重叠的加法
    winGain = len2 / sum(win)
    # winGain


    # 噪声大小计算
    nFFT = 2 * 2 ** (nextpow2.nextpow2(len_))
    noise_mean = np.zeros(nFFT)
    # nFFT, noise_mean


    j = 0
    for k in range(1, 6):
        noise_mean = noise_mean + abs(np.fft.fft(win * x[j:j + len_], nFFT))
        j = j + len_
    noise_mu = noise_mean / 5
    # noise_mean, noise_mu


    # 分配内存初始化变量
    k = 1
    img = 1j
    x_old = np.zeros(len1)
    Nframes = len(x) // len2 - 1
    xfinal = np.zeros(Nframes * len2)


    # Start
    for n in range(0, Nframes):
        # Windowing
        insign = win * x[k - 1:k + len_ - 1]
        # compute fourier transform of a frame
        spec = np.fft.fft(insign, nFFT)
        # compute the magnitude
        sig = abs(spec)

        # 保存噪声相位信息
        theta = np.angle(spec)
        #     print( x[k-1:k + len_ - 1])
        #     print(insign)
        SNRseg = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)
        #     print(SNRseg)

        if Expnt == 1.0:  # 幅度谱
            alpha = berouti1(SNRseg)
        else:  # 功率谱
            alpha = berouti(SNRseg)
        #############
        sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt;
        # 当纯净信号小于噪声信号的功率时
        diffw = sub_speech - beta * noise_mu ** Expnt
        # beta negative components

        z = find_index(diffw)
        if len(z) > 0:
            # 用估计出来的噪声信号表示下限值
            sub_speech[z] = beta * noise_mu[z] ** Expnt
            # --- implement a simple VAD detector --------------
            if SNRseg < Thres:  # Update noise spectrum
                noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt  # 平滑处理噪声功率谱
                noise_mu = noise_temp ** (1 / Expnt)  # 新的噪声幅度谱
            # flipud函数实现矩阵的上下翻转，是以矩阵的“水平中线”为对称轴
            # 交换上下对称元素
            sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])
            x_phase = (sub_speech ** (1 / Expnt)) * (
                        np.array([math.cos(x) for x in theta]) + img * (np.array([math.sin(x) for x in theta])))
            # take the IFFT

            xi = np.fft.ifft(x_phase).real
            # --- Overlap and add ---------------
            xfinal[k - 1:k + len2 - 1] = x_old + xi[0:len1]
            x_old = xi[0 + len1:len_]
            k = k + len2


    # 保存文件
    wf = wave.open('outfile.wav', 'wb')
    # 设置参数
    wf.setparams(params)
    # 设置波形文件 .tostring()将array转换为data
    wave_data = (winGain * xfinal).astype(np.short)
    # print(type(wave_data))
    wf.writeframes(wave_data.tostring())
    wf.close()

    return params