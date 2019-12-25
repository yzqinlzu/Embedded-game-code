import reduction, split, Remodel
from split_by_silence import my_split2
######识别调用的接口
######识别调用的接口
######识别调用的接口
def strat(path):
    """
    识别降噪，分割总接口
    :param path: 要识别的WAV文件路径
    :return: 识别结果列表
    """
    # par = reduction.myreduction(path) # 降噪算法正在继续优化 暂停使用
    # n_list = split.mysplit(path)
    n_list = my_split2(path)
    cn={'W':'愤怒','L':'无聊','E':'厌恶','A':'恐惧','F':'快乐','T':'悲伤','N':'中性'}
    j=0
    ans=[]
    for i in n_list:
        ans.append(cn[Remodel.wav_predict("./save_audio/"+str(j)+".wav")])
        j+=1
    return  ans