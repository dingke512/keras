import pickle
import re
import jieba
from keras.models import load_model
import numpy as np
from keras_preprocessing.sequence import pad_sequences
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class SentimentAnalysis():
        def __init__(self):
            pass
        # 清洗数据，正则
        def clear(self,text):
            r3 = "[.,!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\s\\\\，；。=？、：“”‘’《》【】￥……（）]+"
            result=re.sub(pattern=r3,repl='',string=text)
            return result

        def data_cut(self,data):
            # 分词
            for i in range(len(data)):
                data[i] = self.clear(data[i])
                data[i] = list(jieba.cut(data[i]))
                # print(data[i])
            return data
        def tokenize_load(self, data):
            # 构建词典，将每条数据的词替换成索引
            with open('tokenizer.pickle', 'rb') as handle:
                t1 = pickle.load(handle)

            sequences = t1.texts_to_sequences(self.data_cut(data))
            max_len = 30
            data = pad_sequences(sequences,maxlen=max_len)
            return data
        def modeload(self):
            # 模型加载
            model = load_model("weibo_senti.h5")
            np.set_printoptions(suppress=True)
            return model
        def res(self,data):
            # 预测
            print("预测")
            data = self.tokenize_load(data)
            self.model = self.modeload()
            result = self.model.predict(data)
            print(result)

if __name__ == '__main__':

    data = ['[鼓掌]//@慕春彦: 一流的经纪公司是超模的摇篮！[鼓掌] //@姚戈:东方宾利强大的名模军团！',
            '好感动[亲亲]大家都陆陆续续收到超极本尼泊尔的奖品了，没想到你还带着去看瓷房子~祝蜜月快乐哦'
             ,'烦死了，又要加班了，我都要哭了呀',
            ]

    sa = SentimentAnalysis()
    sa.res(data)




