import re
import pandas as pd
import jieba
from keras.layers import Dense,Bidirectional,Embedding,LSTM,GRU
import matplotlib.pyplot as plt
from keras_preprocessing.sequence import pad_sequences
from keras import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from keras_preprocessing.text import Tokenizer
import tensorflow as tf
from keras.optimizers import SGD, RMSprop

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# pandas读取数据
df = pd.read_csv('weibo_senti_100k.csv',encoding='utf-8')
# 划分data与label
data = df['review']
label = df['label']
# 清洗数据，正则
def clear(text):
    r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\s\\\\，；。=？、：“”‘’《》【】￥……（）]+"
    result=re.sub(pattern=r3,repl='',string=text)
    return result

df['words'] = data.apply(lambda x: clear(str(x)))
# print(df['words'])
# 分词
data = df['words'].apply(lambda x: list(jieba.cut(x)))
data = data.tolist()

# 构建词典，将每条数据的词替换成索引
t1 = Tokenizer(num_words=10000)
t1.fit_on_texts(data)
# 将文本替换索引
sequences = t1.texts_to_sequences(data)
word_index = t1.word_index
data = pad_sequences(sequences, maxlen=30)  # 对整型序列进行填充（截断），只取前100个单词
label = np.asarray(label)
print(type(data))
# 保存词典
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(t1, handle, protocol=pickle.HIGHEST_PROTOCOL)
# 拆分数据集
train_x,test_x,train_y,test_y = train_test_split(data,label,test_size=0.3,random_state=5)
print(type(train_x))
print(train_x)

# # 将序列反转
train_x = [i[::-1] for i in train_x]
test_x = [i[::-1] for i in test_x]
#  填充序列,并将数据转为numpy.ndarray(多维度数组)
max_len=30
train_x = pad_sequences(train_x, maxlen=max_len)
test_x = pad_sequences(test_x, maxlen=max_len)
print(train_x.shape)
print(type(train_x))
# 模型设计
model = Sequential()
# 使用embedding层学习词嵌入
model.add(Embedding(input_dim=10000, output_dim=64,))
# 双向LSTM
model.add(Bidirectional(GRU(units=64)))
model.add(Dense(1, activation="sigmoid"))
model.summary()
# 模型的编译
model.compile(optimizer=RMSprop(), loss="binary_crossentropy", metrics=["accuracy"])
# 训练模型
his = model.fit(x=train_x, y=train_y, batch_size=128, epochs=2, validation_data=(test_x,test_y),verbose=0)
d1=his.history
est_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print('Test accuracy:', test_acc)
# 训练数据绘图
plt.rcParams['font.sans-serif']=['SimHei']
# plt.ylim(0.5,1)
plt.plot(d1.get("val_accuracy"),color="green", label="val_acc")
plt.plot(d1.get("accuracy"), color="red", label="acc")
plt.legend()
plt.show()
model.save('weibo_senti.h5')

