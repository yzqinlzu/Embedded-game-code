from pydub import AudioSegment
from pydub.silence import split_on_silence


def my_split2(path):

    sound = AudioSegment.from_mp3(path)
    loudness = sound.dBFS
    # print(loudness)

    chunks = split_on_silence(sound,
                              # must be silent for at least half a second,沉默半秒
                              min_silence_len=430,

                              # consider it silent if quieter than -16 dBFS
                              silence_thresh=-45,
                              keep_silence=400

                              )
    print('总分段：', len(chunks))

    # 放弃长度小于2秒的录音片段
    for i in list(range(len(chunks)))[::-1]:
        if len(chunks[i]) <= 1000 or len(chunks[i]) >= 10000:
            chunks.pop(i)
    print('取有效分段(大于1s小于10s)：', len(chunks))

    '''
    for x in range(0,int(len(sound)/1000)):
        print(x,sound[x*1000:(x+1)*1000].max_dBFS)
    '''
    pathdir=[]
    for i, chunk in enumerate(chunks):
        chunk.export("save_audio2/{0}.wav".format(i), format="wav")
        pathdir.append("save_audio2/{0}.wav".format(i))
        # print(i)
    return pathdir

# my_split2('E:\\PycharmProjects\\flasktest\\upload\\flietopredict.wav')