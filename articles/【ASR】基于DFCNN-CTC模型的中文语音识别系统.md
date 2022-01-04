**写在前面：**时间是让人猝不及防的东西，恍惚间半年就过去了，笔者这半年工作太忙，学习近乎停滞，回想起来懊悔不已，半年时间浪费了，从这个月开始要振作起来，无论多忙都要保持一颗求学上进的心。及时当勉励，岁月不待人！

## 1 背景介绍

笔者前段时间停车充电的时候看了李宏毅的 NLP 课程，讲的很生动，想起来之前笔者也学习过语音识别的一些浅显知识，觉得有必要总结一下，于是有了这篇文章，本文主要介绍模型结构和具体实现细节，关于 ASR（Automatic Speech Recognition）的发展背景等情况不再赘述。

本文搭建了一个完整的中文语音识别系统，包括声学模型和语言模型，能够将输入的音频信号识别为汉字。本文主要从以下四个方面进行介绍：**数据集、模型结构、音频信号处理、实践环节**。

完整的中文语音识别系统包括两个模型：将语音信号转换为“拼音+声调”的**声学模型**，将“拼音+声调”转换为中文文字的**语言模型**。为什么用两个模型？直接一个模型从语音信号到文字，硬train一发不好吗？如果只是一个孤立词识别的语音识别系统，我相信硬train一发应该也能效果不错，但是对于句子级的语音识别任务，除了单纯地识别出发音元素以外，还需要识别出语音的含义转化为一串中文文字，这个过程复杂度更高，一个模型效果通常不太好。所以通常我们需要声学模型将声学和发音学的知识进行整合，将语音信号经过特征处理后作为声学模型的输入，经过转换后输出可变长的发音元素，中文来讲就是“拼音+声调”。然后我们需要一个语言模型学习“拼音+声调”与文字之间的相互关系，将“拼音+声调”作为语音模型的输入，输出中文文字。本文声学模型使用科大讯飞提出的DFCNN深度全序列卷积神经网络，语言模型则使用transformer模型搭建拼音序列生成汉字序列系统。

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5qH9tenE7VPa6cIYCIzGJIYgyOiao2q7j6PuqVzkOEDhmARhQQNlwO17IbLFj24ZpibKDl8KA1Ceo1g/0?wx_fmt=png)

<center><font face="黑体" size=3>图1 声学模型的任务</font></center>

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5qH9tenE7VPa6cIYCIzGJIYfYj0HO6a4qPH0Pkf1y84wDv4sfibW4LwJF0icXV8jqxC5U5icpH10NL6Q/0?wx_fmt=png)

<center><font face="黑体" size=3>图2 语言模型的任务</font></center>

## 2 数据集

本文数据集采用 [OpenSLR](http://www.openslr.org/resources.php) 上下载的中文免费数据，包括：thchs-30、aishell、primewords、st-cmd 四个数据集，训练集总计大约450个小时。数据标签整理在`data`路径下，包含了解压后数据的路径，以及训练所需的数据标注格式，其中 primewords、st-cmd 目前未区分训练集和测试集，需要手工区分，各个数据集的规模如下：

|    Name    | train  |  dev  | test |
| :--------: | :----: | :---: | :--: |
|  aishell   | 120098 | 14326 | 7176 |
| primewords | 40783  | 5046  | 5073 |
|  thchs-30  | 10000  |  893  | 2495 |
|   st-cmd   | 10000  |  600  | 2000 |

数据集总共约`40G`，无法一次性加载进内存（笔者电脑配置不够），放在磁盘每个 epoch 都去磁盘里面执行 wave 读取操作的话，速度又太慢。这里我将所有语音文件和对应的数据标签全部存入 MongoDB 数据库，需要的时候再去调用。

### 2.1 音频数据读取

这里使用 Python 自带的标准库 wave 读取音频数据，具体函数如下：

```python
import wave

def read_wave(path):
    with wave.open(path, "rb") as f:
        params = f.getparams()
        # 通道数，量化位数（byte），采样率，采样点数
        nchannels, sampwidth, framerate, nframes = params[:4]
        str_data = f.readframes(nframes)
    return nchannels, sampwidth, framerate, nframes, str_data
```

以一个语音文件例子 `wav_path = "./dst.wav"` 为例，`nchannels, sampwidth, framerate, nframes` 分别为 `(1, 2, 8000, 1043520)`，分别表示单通道、16bit、8k、1043520个采样点等基本属性。`str_date` 为 `bytes` 类型，前20个字符分别为 `b'*\x00\x14\x00\xe0\xff\x8c\xffO\xff\x18\xff\x08\xff1\xffo\xff\xd8\xff'` 。

```python
print(nchannels, sampwidth, framerate, nframes)
# (1, 2, 8000, 1043520)

print(type(str_data), str_date[:20])
# (bytes, b'*\x00\x14\x00\xe0\xff\x8c\xffO\xff\x18\xff\x08\xff1\xffo\xff\xd8\xff')
```

可以看到`str_date` 为 `bytes` 类型，那么如何获取具体数字信号呢？可以使用 `np.frombuffer` 函数：

```python
import numpy as np

wave_data = np.frombuffer(str_data, dtype = np.short)
wave_data, wave_data.shape
# (array([  42,   20,  -32, ...,  844, 1339, 1805], dtype=int16), (1043520,))
```

`wave_data` 是一个包含 `104350` 个元素的一维数组，元素数据类型为 `int16` ，范围在 $2^{15} \thicksim 2^{15}-1$ 之间。

plot出来看一下。如下图，横轴表示采样点，纵轴表示声音信号振幅。

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5oMo41JTGH6sIorjhr3K8pOpUiaO7qLoqic0J2M9WiaGOmBL0icxQGK3GSD6m01at2xnVbibS5HbZichicFA/0?wx_fmt=png)

<center><font face="黑体" size=3>图3 声音信号波形图</font></center>

### 2.2 音频数据写入MongoDB

MongoDB 是由C++语言编写的，是一个基于分布式文件存储的开源数据库系统。MongoDB 将数据存储为一个文档，数据结构由键值`key: value`对组成。MongoDB 文档类似于 `JSON` 对象，文档字段值可以包含其他文档，数组及文档数组。

类似关系型数据库中“表（table）”的概念，MongoDB中对应的是“集合（collection）”，这里我们根据四个数据集，创建4个`collection` ，分别将数据集插入对应集合。Python 中提供了与MongoDB数据库交互的第三方库 `pymongo` ，这里我们使用 `pymongo`进行写入操作。

```python
from pymongo import MongoClient

class WriteWaveToMongoDB:
    def __init__(self, dir_path, data_param, db):
        self.dir_path = dir_path
        self.data_param = data_param
        self.db = db
    
    def get_wave_list(self, data_name):
        """
        从4个数据集的txt文件中获取语音文件名、拼音、汉字等3个列表
        """
        # 语音文件名列表
        wav_lst = []
        # 拼音列表
        pny_lst = []
        # 汉字列表
        han_lst = []
        # 拼接文件路径
        file_name = self.dir_path + self.data_param.get(data_name)
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                # 数据集中语音文件名、拼音、汉字等3个字段是用制表符分割的
                wav_file, pny, han = line.strip().split('\t')
                if wav_file and pny and han:
                    wav_lst.append(self.dir_path + wav_file)
                    pny_lst.append(pny.split())
                    han_lst.append(han.replace(" ", ""))
        return wav_lst, pny_lst, han_lst
    
    def writing_data(self):
        """
        将每组音频数字信号、拼音、汉字组成一个文档，插入MongoDB中
        """
        for data_name in self.data_param:
            print("processing {}".format(data_name))
            # 获取数据库中对应的collection
            collection = self.db[data_name]
            wav_lst, pny_lst, han_lst = self.get_wave_list(data_name)
            for i in trange(len(wav_lst)):
                item = {"id": i, 
                        "wav":{}, 
                        "pny":pny_lst[i], 
                        "han":han_lst[i],
                       }
                nchannels, sampwidth, framerate, nframes, str_data = read_wave(wav_lst[i])
                item["wav"]["nchannels"] = nchannels
                item["wav"]["sampwidth"] = sampwidth
                item["wav"]["framerate"] = framerate
                item["wav"]["nframes"] = nframes
                item["wav"]["str_data"] = str_data
                # 插入一条数据
                collection.insert_one(item)


if "__name__" == "__main__":
    data_param = {
        "thchs30":"thchs_train.txt",
        "aishell":"aishell_train.txt",
        "prime":"prime.txt",
        "stcmd":"stcmd.txt",
    }
    data_path = './data/'
    db = MongoClient(host="127.0.0.1", port=27017)["asr"]
    write2mongo = WriteWaveToMongoDB(data_path, data_param, db)
    write2mongo.writing_data()
```

插入完成后，以 `thchs30` 这个collection为例，我们取一条数据看下数据结构：

```python
db['thchs30'].find_one({"id":1})

"""
{
	'_id': ObjectId('5e201377a070db560e390b54'),
 	'han': '他仅凭腰部的力量在泳道上下翻腾蛹动蛇行状如海豚一直以一头的优势领先',
 	'id': 1,
 	'wav': {
 			'nchannels': 1,
  			'str_data': b'Y\x00Y\x00d\x00a\x00V\x00L\x00G\...,
  			'nframes': 141440,
  			'framerate': 16000,
  			'sampwidth': 2
  		   },
	'pny': ['ta1','jin3','ping2','yao1','bu4','de','li4','liang4','zai4','yong3','dao4','shang4','xia4',
'fan1','teng2','yong3','dong4','she2','xing2','zhuang4','ru2','hai3','tun2','yi4','zhi2','yi3','yi1','tou2','de','you1','shi4','ling3','xian1']
}
  
""" 
```

## 3 模型结构

### 3.1 声学模型

科大讯飞在2016年提出了一种全新的语音识别框架，称为全序列卷积神经网络（deep fully convolutional neural network，DFCNN）。DFCNN 将一句语音转化成一张图像作为输入，把时间和频率作为图像的两个维度（语谱图），通过较多的卷积层和池化(pooling)层的组合，实现对整句语音的建模，输出单元则直接与最终的识别结果（比如音节或者汉字）相对应。利用 CNN 的参数共享机制，可以将参数数量下降一个级别，且深层次的卷积和池化层能够充分考虑语音信号的上下文信息，且可以在较短的时间内就可以得到识别结果，具有较好的实时性。

DFCNN 模型结构如下：

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5oMo41JTGH6sIorjhr3K8pOze14CiciarHesXNicq38qR8qpsMPKlOuLHgbY2TraiczoicXjZOib1JY0XuQ/0?wx_fmt=png)

<center><font face="黑体" size=3>图4 DFCNN模型结构示意图</font></center>

由图4可见，DFCNN 模型输入是一个声音信号的语谱图（Spectrogram），那么什么是语谱图，语谱图怎么求？这点我们在下节介绍。输入层紧接着的是N个由卷积层、归一化层、池化层组成的卷积神经网络结构，也是模型的主要组成单元，这里我们称为**卷积单元**。N个卷积单元之后接了一个 `Reshape` 层，这一层可以将上一层的输出由 4 维转换为 3 维，即从 `[batch_size, seq_len, width, depth]` 转换为 `[batch_size, seq_len, width* depth]` 然后就是两层全连接神经网络，模型训练时中间夹了 `Dropout` 用于缓解过拟合问题。

#### 3.1.1 卷积单元

**卷积层：**每个卷积单元里包含两个卷积层，卷积核大小都为 `3*3` ，padding 方式为 `same` ，激活函数设置为 `relu` ，卷积核数量为32、64、128等。

**归一化层：**这里使用 `BatchNormlization`，在一个 `batch` 内从 `length, width, depth` 三个方向对输入向量进行标准化，这样做可以使神经网络每一层的输入基本分布在一个标准差和均值下，**目的就是让每一层的分布稳定下来**，让后面的层可以在前面层的基础上安心学习知识，解决深层网络下的梯度消失问题。

**池化层：**使用 `Maxpooling` 方式，池化尺寸为 `2*2` ，步长和池化尺寸一致，padding方式为 `valid` ，所以池化之后尺寸减半。这一层作用是减小输出向量尺寸大小和缓解过拟合。

```python
class CNNCell(keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), activation='relu', pool=True, **kwargs):
        super(CNNCell, self).__init__(**kwargs)
        self.supports_masking = True
        self.conv2d_1 = keras.layers.Conv2D(filters,
                                            kernel_size,
                                            use_bias=True,
                                            activation=activation,
                                            padding='same',
                                            kernel_initializer='he_normal', )
        self.conv2d_2 = keras.layers.Conv2D(filters,
                                            kernel_size,
                                            use_bias=True,
                                            activation=activation,
                                            padding='same',
                                            kernel_initializer='he_normal', )
        self.batch_norm_1 = keras.layers.BatchNormalization(axis=-1)
        self.batch_norm_2 = keras.layers.BatchNormalization(axis=-1)
        self.pool = pool
        self.max_pool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid")
        self.mask_pool = keras.layers.MaxPool2D(pool_size=(2, 1), strides=None, padding="valid")

    def call(self, inputs, mask=None, **kwargs):
        x = self.conv2d_1(inputs)
        x = self.batch_norm_1(x)
        x = self.conv2d_2(x)
        x = self.batch_norm_2(x)
        if mask is not None:
            # mask:(batch_size, seq_len_k)
            mask = tf.cast(mask, tf.float32)
            # (batch_size, seq_len_k) => (batch_size, seq_len_k, 1, 1)
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = tf.expand_dims(mask, -1)
            x *= mask
        if self.pool:
            x = self.max_pool(x)
            mask = self.mask_pool(mask)
        return x, mask
```

#### 3.1.2 CTC

对于语音识别来说，训练数据的输入是一段音频，输出是它转录的文字（transcript），但是我们是不知道字母和语音是怎么对齐（align）的。这使得训练语音识别比看起来更加复杂。

要人来标注这种对齐是非常困难而且容易出错的，因为很多音素的边界是很难区分，如下图5，人通过看波形或者频谱是很难准确的区分其边界的。之前基于 HMM 的语音识别系统在训练声学模型是需要对齐，我们通常会让模型进行强制对齐（forced alignment）。类似的在手写文字识别中，也会存在同样的问题，虽然看起来比声音简单一些，传统的手写文字识别方法首先需要一个分割（segmentation）算法，然后再识别。

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5p6gAN1APvD1WCdCKiaSgPO16COGag9wn9J2qzYIFdtBDCuFZ9UVV9Iz2XyYFTUN1XRdMXxkwT7Rag/0?wx_fmt=png)

<center><font face="黑体" size=3>图5 文字识别和语音识别中的的对齐问题</font></center>

**CTC（Connectionist Temporal Classification）算法并不要求输入输出是严格对齐的**。

为了更好的理解CTC的对齐方法，先举个简单的对齐方法。假设对于一段音频，我们希望的输出是 $Y = [c,a,t]$ 这个序列，一种将输入输出进行对齐的方式如下图6所示，先将每个输入对应一个输出字符，然后将重复的字符删除。
![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5p6gAN1APvD1WCdCKiaSgPO1oudcpvHgLP9BHptaP9pR3hgozooH09p7Py2erDloxna87C7fhicHGrA/0?wx_fmt=png)

<center><font face="黑体" size=3>图6 cat序列对齐</font></center>

仔细观察可以发现，上述对齐方式有两个问题：

- 通常这种对齐方式是不合理的。比如在语音识别任务中，有些音频片可能是无声的，这时候应该是没有字符输出的。
- 对于一些本应含有重复字符的输出，这种对齐方式没法得到准确的输出。例如输出对齐的结果为 $[h,h,e,l,l,l,o]$，通过去重操作后得到的不是“hello”而是“helo“，

为了解决上述问题，CTC算法引入的一个新的占位符用于输出对齐的结果。这个占位符称为空白占位符，通常使用符号 $\epsilon$，也称 blank。这个符号在对齐结果中输出，但是在最后的去重操作会将所有的 $\epsilon$ 删除得到最终的输出。利用这个占位符，可以将输入与输出有了非常合理的对应关系，如下图7所示。

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5p6gAN1APvD1WCdCKiaSgPO1lWuWpQ0pYueQbx8icSu3pXG53KdjHtSbmyNqYDUSnwmPIqLIviadaCNQ/0?wx_fmt=png)

<center><font face="黑体" size=3>图7 CTC对齐示意</font></center>

在这个映射方式中，如果在标定文本中有重复的字符，对齐过程中会在两个重复的字符当中插入 $\epsilon$ 占位符。利用这个规则，上面的“hello”就不会变成“helo”了。
了解了 CTC 原理我们就可以来计算损失函数了，具体公式就不在这里推导了，具体可参考知乎文章：[详解CTC](https://zhuanlan.zhihu.com/p/42719047)。这里我们直接使用 tensorflow 中的API进行计算即可。详细代码如下：

```python
class CtcBatchCost(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CtcBatchCost, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        labels, y_pred, input_length, label_length = inputs
        y_pred = y_pred[:, :, :]
        return keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], 1
```

`ctc_batch_cost(y_true, y_pred, input_length, label_length)` 函数参数介绍如下：

- y_true：形如 `[batch_size, max_string_length]` 的张量，包含标签的真值和padding值

- y_pred：形如 `[batch_size, time_step, num_categories]` 的张量，包含预测值或输出的 softmax 值

- input_length：形如 `[batch_size,1]` 的张量，y_pred 中未进行补零操作的长度，也就是原始 input_data 未补零的长度除以MaxPooling 缩放的倍数

- label_length：形如`[batch_size,1]` 的张量，包含 y_true 中每个 batch 标签序列 padding 前的实际长度

#### 3.1.3 模型整体结构和代码

声学模型 DFCNN 代码如下：

```python
class DFCNN(object):
    """a dfcnn network for Amodel."""

    def __init__(self, vocab_size=1296, inp_width=200, lr=0.0008, gpu_nums=1, is_training=True):
        self.vocab_size = vocab_size
        self.gpu_nums = gpu_nums
        self.lr = lr
        self.is_training = is_training
        self.inp_width = inp_width
        self.ctc_batch_cost_layer = CtcBatchCost(name='ctc')
        self.dense_1 = keras.layers.Dense(256, activation="relu",
                                          use_bias=True,
                                          kernel_initializer='he_normal',
                                          name="dense_1")
        self.dense_2 = keras.layers.Dense(vocab_size, activation='softmax',
                                          use_bias=True, kernel_initializer='he_normal',
                                          name="dense_2")
        self._model_init()
        if self.is_training:
            self._ctc_init()
            self.opt_init()

    def _model_init(self):
        self.inputs = keras.layers.Input(name='the_inputs', shape=(None, self.inp_width))
        x = keras.layers.Masking(mask_value=0.0)(self.inputs)
        mask = keras.layers.Lambda(lambda x: x._keras_mask, name="init_mask")(x)
        x = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=-1), name="expand_dims")(x)
        x, mask = CNNCell(32)(x, mask=mask)
        x, mask = CNNCell(64)(x, mask=mask)
        x, mask = CNNCell(128)(x, mask=mask)
        x, mask = CNNCell(128, pool=False)(x, mask=mask)
        x = keras.layers.Reshape((-1, self.inp_width // 8 * 128), name="reshape")(x)
        x = keras.layers.Dropout(rate=0.2)(x, training=self.is_training)
        x = self.dense_1(x)
        x = keras.layers.Dropout(rate=0.2)(x, training=self.is_training)
        self.outputs = self.dense_2(x)
        if mask is not None:
            self.outputs = keras.layers.Lambda(lambda x:x[0] * x[1][:, :, :, 0])([self.outputs, mask])
        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs)

    def _ctc_init(self):
        self.labels = keras.Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = keras.Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = keras.Input(name='label_length', shape=[1], dtype='int64')
        self.loss_out = self.ctc_batch_cost_layer([self.labels, self.outputs, self.input_length, self.label_length])
        self.ctc_model = keras.Model(inputs=[self.inputs, self.labels, self.input_length, self.label_length],
                                     outputs=self.loss_out, name="ctc_model")

    def opt_init(self):
        opt = keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.01, epsilon=10e-8)
        # if self.gpu_nums > 1:
        #     self.ctc_model = keras.utils.multi_gpu_model(self.ctc_model,gpus=self.gpu_nums)
        # keras自定义损失函数要求函数前两个参数分别为y_true, y_pred
        self.ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt)
```

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5qszvpibkSkzicib5IkvBWoItBm9ToAb70C0SQjl6K737mT0BeI95lkMZgUWFk6CxYxDjTIQicricJXKbA/0?wx_fmt=png)

<center><font face="黑体" size=3>图8 DFCNN-CTC的模型结构</font></center>

### 3.2 语言模型

前面讲过语言模型的任务是将发音单元转换为文字，具体就是把“拼音+声调”转换为一串文字，从图2可以看出，这里输入和输出一一对应，其实就是一个序列标注任务，关于序列标注任务的模型有很多，这里我们选用 **Transformer** 模型。

Transformer 模型来自于 Google 提出的一篇论文 **Attention Is All You Need** 。论文中提出一个全新的模型，叫 Transformer，抛弃了以往深度学习任务里面使用到的 CNN 和 RNN ，取而代之的是一种 self-Attention 的结构，将Attention思想发挥到了极致。目前大热的预训练模型 **Bert** 就是基于 Transformer 构建的，这个模型广泛应用于NLP领域，例如机器翻译，问答系统，文本摘要和语音识别等等方向。原版 Transformer 模型的结构如下：

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5pLns3OIaoCWelJic9jx2uFw4x99iaaPt0JLPCpfF9yTSWU8rocqoOGDYsnIbMII1KevH2hvHBDnnlg/0?wx_fmt=png)

<center><font face="黑体" size=3>图9 原版Transfomer模型结构图</font></center>

从图9可以看出，原版模型是一个标准的 sequence to sequence 模型架构，左侧是 Encoder 模型，右侧是 Decoder 模型，因为在提出时它解决的也是一个 seq2seq 的问题，比如机器翻译、文本摘要、问答等等。这里我们由于是一个序列标注的任务，所以只使用 Encoder 部分就足够了。可以看到 Encoder 部分主要分为**位置编码（Position Encoding）、多头注意力机制（Multi-Head Attention）、残差结构（Add & Norm)、前馈神经网络（Point-wise FFN）**等4个主要模块，下面进行详细介绍。

#### 3.2.1 self-Attention

**self-Attention** 是 Transformer 模型的最核心的机制，几乎可以用 **self-Attention** 来指代 Transformer 模型了。那么什么是 **self-Attention**？self-Attention 通常是基于多头注意力机制（Multi-Head Attention）实现的，就是把多头注意力机制的三个输入参数`Query, Key, Value` 全部赋值为一个向量，那么对于这个向量来说就是自己和自己做 Attention 计算，所以叫 **self-Attention** 。

那么问题来了，什么叫 Multi-Head Attention？要了解这个问题，**先要搞清楚什么是 Attention**。Attention 分成很多种，根据计算机制可以分成乘性Attention、加性Attention等等，根据注意力的范围又可以分成全局Attention、局部Attention、随机Attention、混合Attention等等，这几年已经被人玩出花来了，这里就不一一介绍，只讲解使用最广的 **Scaled Dot-Product Attention**。

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5rC4JSMf3m2tncuTTRjZjjMPJxCAmxJ0HslxWFQJpXrXN7icsic82TFcOOO97OWsCVyxrmbNyL6ItZg/0?wx_fmt=png)

<center><font face="黑体" size=3>图10 Scaled Dot-Product Attention结构图</font></center>

从上图10可以看出，Scaled Dot-Product Attention 模块有3个输入`Q, K, V` ，也就是我们前面说的查询、键、值，`Q`代表当前我们关注的向量通常就是上一层的输出，`K` 和 `V`是一对组合，`K`表示注意力范围内的序列集合，`V`表示注意力范围内的序列对应的隐层向量，通常情况下我们直接 `K = V`。Attention 通常用于 seq2seq 架构，用于捕捉解码器每一个单元与编码器输出序列的关系，所以在 seq2seq 架构中`Q`就是解码器中上一层的输出，`K` 和 `V`是编码器的输出。而在 self-Attention 中，前面说过`Q, K, V`直接都是上一层的输出。

那么具体在 Attention 中`Q, K, V`怎么计算的呢？下面给出一个公式
$$
\begin {equation}
Attention(Q,K,V) = softmax_k(\frac {QK^T}{\sqrt {d_k}}) V
\label {eq:3.1}
\end {equation}
$$
假设`Q` 的形状为`[batch_size, Tq, dim]`, `V` 的形状为 `[batch_size, Tv, dim]` and `K` 的形状为 `[batch_size, Tv, dim]`，这个形状应该怎么理解呢？第一个维度 `batch_size` 是深度学习里面的一个常规的概念表示 `批的大小` ，说人话就是一次计算要处理多少个样本；第二个维度 `Tq和Tv` 就表示查询和键的序列长度，简单来说就是这个句子有多少词，每个词要跟编码器多少个词进行注意力交互；第三个维度就是张量的深度，简单来说就是每个词向量的长度。那么Attention的计算可分为以下三步：

- 把 `Q` 和 `K` 矩阵相乘，`matmul(Q, K, transpose_b=True)`，然后按深度的平方根因子 $\sqrt {d_k}$ 进行缩放，得到一个 `[batch_size, Tq, Tv]`的矩阵。这一步有啥意义？从操作来看是把 `Q` 矩阵序列中的每一个词向量与 `V` 矩阵序列中的每一个词向量两两相乘再除以一个系数，是不是有点熟悉，对，就是余弦相似度！（除了系数有些差别，含义是一样的）所以第一步得到的就是一个相似度矩阵 `scores`。
- 在最后一个维度上进行 `softmax`归一化 ，`softmax(socres, axis=-1)`。这一步的目的很简单，就是把相似度矩阵归一化。比如相似度矩阵某一行为 `[0.3, 0.05, 5]`，归一化后变为 `[0.00895047, 0.00697063, 0.9840789]`，权重之和为1，这一步的结果记为 `distribution`。
- 把 `distribution`与`V`矩阵相乘，`matmul(distribution, value)`。这一步怎么理解呢？我们知道`distribution`的一行表示`Q`中的某个向量`q` 与 `K` 中 `Tv` 个向量的相似度，在乘以 `V`，实际上相当于把`V`中的`Tv` 个向量按相似度加权求和得到一个向量。

简而言之，Attention 机制就是结合注意力范围序列 `K` 对 `Q` 中包含的信息进行增强和抑制，通过更加关注与输入的元素相似的部分，抑制其它无用的信息。其最大的优势就是能一步到位的考虑全局联系和局部联系，且能并行化计算，这是 CNN 和 RNN 无法做到的。

这里推荐一篇文章[浅谈 Attention 机制的理解](https://www.cnblogs.com/ydcode/p/11038064.html#%E4%BB%80%E4%B9%88%E6%98%AF%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6)。

**什么是 Multi-Head Attention **？多头注意力机制，就是多做几组 Attention，有多少 head，就有多少组，然后把结果拼接起来。比如原文做了8组，如下图：

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5rC4JSMf3m2tncuTTRjZjjMCqibiacu11j2pnQ2rIpmonn7aZS015DONEKhk6e9wneWJiavUq2phjMTw/0?wx_fmt=png)

<img src="https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5rC4JSMf3m2tncuTTRjZjjMBVWfmF4krx7a23SZPuO0X5NVfAticZ4qsTibzKibSBUbUgAGIvCSicGibRA/0?wx_fmt=png" style="zoom: 50%;" />

<center><font face="黑体" size=3>图11 多头注意力机制</font></center>

具体代码实现如下：

```python
class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, latent_dim, heads, mask_right=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.heads = heads
        self.mask_right = mask_right
        assert self.latent_dim % heads == 0, "the latent_dim: {} must can be divisible by heads!"
        self.depth = self.latent_dim // heads
        self.q_dense = keras.layers.Dense(self.latent_dim, use_bias=False)
        self.k_dense = keras.layers.Dense(self.latent_dim, use_bias=False)
        self.v_dense = keras.layers.Dense(self.latent_dim, use_bias=False)

    def call(self, inp, **kwargs):
        q, k, v = inp[:3]
        v_mask, q_mask = None, None
        if len(inp) > 3:
            # x_mask: [batch_size, seq_len_x]
            v_mask = inp[3]
            if len(inp) > 4:
                q_mask = inp[4]
        wq = self.q_dense(q)
        wk = self.k_dense(k)
        wv = self.v_dense(v)
        # (batch_size, seq_len, latent_dim) =>(batch_size, seq_len, heads, depth)
        wq = tf.reshape(wq, (tf.shape(wq)[0], tf.shape(wq)[1], self.heads, self.depth))
        wk = tf.reshape(wk, (tf.shape(wk)[0], tf.shape(wk)[1], self.heads, self.depth))
        wv = tf.reshape(wv, (tf.shape(wv)[0], tf.shape(wv)[1], self.heads, self.depth))
        # (batch_size, seq_len, heads, depth) => (batch_size, heads, seq_len, depth)
        wq = tf.transpose(wq, perm=(0, 2, 1, 3))
        wk = tf.transpose(wk, perm=(0, 2, 1, 3))
        wv = tf.transpose(wv, perm=(0, 2, 1, 3))
        # => (batch_size, heads, seq_len_q, seq_len_k)
        scores = tf.matmul(wq, wk, transpose_b=True)
        # 缩放因子
        dk = tf.cast(self.depth, tf.float32)
        # scores[:, i, j] means the simility of the q[j] with k[j]
        scores = scores / tf.math.sqrt(dk)

        if v_mask is not None:
            # v_mask:(batch_size, seq_len_k)
            v_mask = tf.cast(v_mask, tf.float32)
            # (batch_size, seq_len_k) => (batch_size, 1, 1, seq_len_k)
            for _ in range(K.ndim(scores) - K.ndim(v_mask)):
                v_mask = tf.expand_dims(v_mask, 1)
            scores -= (1 - v_mask) * 1e9
        # 解码端，自注意力时使用。预测第三个词仅使用前两个词
        if (self.mask_right is not False) or (self.mask_right is not None):
            if self.mask_right:
                # [1,1,seq_len_q,seq_len_k]
                ones = tf.ones_like(scores[:1, :1])
                # 不包含对角线的上三角矩阵，每个元素是1e9
                mask_ahead = (ones - tf.linalg.band_part(ones, -1, 0)) * 1e9
                # 遮掉所有未预测的词
                scores = scores - mask_ahead
            else:
                # 这种情况下，mask_right是外部传入的0/1矩阵，shape=[q_len, k_len]
                mask_ahead = (1 - K.constant(self.mask_right)) * 1e9
                mask_ahead = K.expand_dims(K.expand_dims(mask_ahead, 0), 0)
                self.mask_ahead = mask_ahead
                scores = scores - mask_ahead
        scores = tf.math.softmax(scores, -1)
        # (batch_size, heads, seq_len_q, seq_len_k) => (batch_size, heads, seq_len_q, depth)
        out = tf.matmul(scores, wv)
        # (batch_size, heads, seq_len_q, depth) => (batch_size, seq_len_q, heads, depth)
        out = tf.transpose(out, perm=(0, 2, 1, 3))
        # (batch_size, seq_len_q, heads, depth) => (batch_size, seq_len_q, latent_dim)
        out = tf.reshape(out, (tf.shape(out)[0], tf.shape(out)[1], self.latent_dim))
        if q_mask:
            # q_mask:(batch_size, seq_len_q)
            q_mask = tf.cast(q_mask, tf.float32)
            # (batch_size, seq_len_q) => (batch_size, seq_len_q, 1)
            for _ in range(K.ndim(out) - K.ndim(q_mask)):
                q_mask = q_mask[..., tf.newaxis]
            out *= q_mask
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.latent_dim
```

self-Attention 到这里就结束了，有没有发现一个问题，self-Attention 忽略了一个重要信息，那就是序列中的词是有顺序的！但是 Attention 机制只是简单的计算相似度然后加权求和，这些都和顺序没有关系，序列中词的位置信息被损失掉了。如果解决这个问题呢，就要依靠 Position Encoding 了。

#### 3.2.2 Position Encoding

到目前为止，Transformer 模型中还缺少一种解释输入序列中单词顺序的方法。为了处理这个问题，Transformer 给 Encoder 层和 Decoder 层的输入添加了一个额外的向量 Positional Encoding，维度和 embedding 的维度一样，这个向量采用了一种很独特的方法来让模型学习到这个值，这个向量能决定当前词的位置，或者说在一个句子中不同的词之间的距离。这个位置向量的具体计算方法有很多种，论文中的计算方法如下：
$$
\begin {equation}
PE(pos,2i) = \sin(\frac {pos}{10000^{2i / d_{model}}})
\label {eq:3.2}
\end {equation}
$$

$$
\begin {equation}
PE(pos,2i+1) = \cos(\frac {pos}{10000^{2i / d_{model}}})
\label {eq:3.3}
\end {equation}
$$

其中 `pos` 是指当前词在句子中的位置，`i` 是指向量中每个值的索引，可以看出，在**偶数位置，使用正弦编码，在奇数位置，使用余弦编码**，这里提供一下代码。

```python
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, latent_dim, maximum_position=1000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.maximum_position = maximum_position
        self.latent_dim = latent_dim
        position = np.arange(self.maximum_position).reshape((self.maximum_position, 1))
        d_model = np.arange(self.latent_dim).reshape((1, self.latent_dim))
        angle_rates = 1 / np.power(10000, (2 * (d_model // 2)) / np.float32(self.latent_dim))
        self.angle_rads = position * angle_rates
        # 将 sin 应用于数组中的偶数索引（indices）；2i
        self.angle_rads[:, 0::2] = np.sin(self.angle_rads[:, 0::2])
        # 将 cos 应用于数组中的奇数索引；2i+1
        self.angle_rads[:, 1::2] = np.cos(self.angle_rads[:, 1::2])
        # (1, maximum_position, latent_dim)
        self.pos_encoding = tf.cast(self.angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x, **kwargs):
        self.seq_len = tf.shape(x)[1]
        self.position_encoding = self.pos_encoding[:, :self.seq_len, :]
        return x + self.position_encoding

    def compute_output_shape(self, input_shape):
        return 1, input_shape[1], self.latent_dim
```

代码逻辑很简单，先根据公式计算一个大的 `position_encoding` 矩阵，然后根据实际输入向量的形状取用对应的部分即可。下面我们把 `position_encoding` 矩阵画出来。

```python
import matplotlib.pyplot as plt
%matplotlib inline

posEnc = PositionalEncoding(128, 500)
plt.pcolormesh(posEnc.pos_encoding[0], cmap='RdBu')
plt.ylabel('Position')
plt.xlabel('Depth')
plt.colorbar()
plt.show()
```

<img src="https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5rsw1AoDj2Ykf7N8y79RCLiaVYXibHDcVjXWIYuibick7BtkSgaepbRlgvXcsgC4Hk0sjTaojo8xQBficQ/0?wx_fmt=png" style="zoom:150%;" />

<center><font face="黑体" size=3>图12 position_encoding矩阵</font></center>

#### 3.2.3 Add & Norm

如图9，在 Transformer 中，每一个子层（self-Attetion，Point-wise FFN）之后都会接一个残缺模块，并且有一个 Layer normalization]，写成公式 $LayerNorm(X + SubLayer(X))$ 。简单来说，就是在当前层的输出向量基础上再加上当前层的输入然后进行归一化。残差结构的思想来源于 ResNet 神经网络，主要是为了解决随着神经网络层数增加带来的两个问题：梯度消失和网络退化。

梯度消失问题是由于随着神经网络层数加深，在进行梯度下降优化时，根据反向传播链式求导法则，离输出层越近的层梯度越大，离输出层越远的层梯度越小甚至接近于0，导致远端网络参数很难更新。要解决这个问题有多种方式：一种就是使用残差结构，相当于每一层加一个常数项1（**dh/dx=d(f+x)/dx=1+df/dx**），这样就算原来的导数 df/dx 很小，这时候误差仍然能够有效的反向传播；还有就是把数据送入激活函数之前进行 Normalization（归一化），把输入转化成均值为0方差为1的数据，对输入数据的分布特征进行优化，尽量避免数据都落在激活函数饱和区；另外使用 ReLu 激活函数也能起到很好的效果。

那解决了梯度消失问题是不是网络就能无限叠加了？实验证明，随着网络层数的增加，网络发生了退化（degradation）的现象：随着网络层数的增多，训练集loss逐渐下降，然后趋于饱和，当再增加网络深度的话，训练集loss反而会增大。这并不是过拟合，因为在过拟合中训练loss是一直减小的。也就是说，当网络退化时，浅层网络能够达到比深层网络更好的训练效果。从信息论的角度讲，在前向传输的过程中，随着层数的加深，数据的原始信息会逐层减少，而残差结构的直接映射的加入，保证了下一层的网络一定比上一层包含更多的信息。

这个结构比较简单，代码如下：

```python
# 残差层，这一层的输入包含 x 和 SubLayer(x)
keras.layers.Lambda(lambda x: x[0] + x[1])
# 层归一化，在序列维度上归一化，在NLP领域常用
keras.layers.LayerNormalization(epsilon=1e-6)
```

#### 3.2.4 Point-wise FFN

这层主要是提供非线性变换，注意到在Multi-Head Attention的内部结构中，我们进行的主要都是矩阵乘法（scaled Dot-Product Attention），即**进行的都是线性变换**。而线性变换的学习能力是不如非线性变化的强的，所以Multi-Head Attention的输出尽管利用了Attention 机制，学习到了每个 word 的新 representation 表达，但是这种 representation 的表达能力可能并不强，我们仍然希望可以**通过激活函数的方式，来强化 representation 的表达能力**。这个结构就是两层全连接神经网络，代码如下：

```python
class FeedForward(keras.layers.Layer):
    def __init__(self, d_model, dff=512, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.d_model = d_model
        self.dense_1 = keras.layers.Dense(dff, activation='relu')
        self.dense_2 = keras.layers.Dense(d_model)

    def call(self, inp, **kwargs):
        out = self.dense_1(inp)
        out = self.dense_2(out)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (self.d_model,)
```

#### 3.2.5 模型整体结构和代码

语言模型 Transformer 代码如下：

```python
class Transformer:
    def __init__(self, vocab_size, em_dim, num_layers, is_training=True):
        self.embedding = keras.layers.Embedding(vocab_size, em_dim, mask_zero=True, name="embedding")
        self.position_em = PositionalEncoding(em_dim, name="position")
        self.mask = keras.layers.Lambda(lambda x: x._keras_mask, name="mask")
        self.num_layers = num_layers
        self.is_training = is_training
        self.mha = [MultiHeadAttention(em_dim, heads=8, name="multiheadAttn_{}".format(i)) for i in range(num_layers)]
        self.dropout_em = keras.layers.Dropout(rate=0.1, name="dropout_em")
        self.dropout_attn = [keras.layers.Dropout(rate=0.1, name="enc_{}_dropout".format(i)) for i in
                             range(self.num_layers)]
        self.dropout_ffn = [keras.layers.Dropout(rate=0.1, name="ffn_{}_dropout".format(i)) for i in
                            range(self.num_layers)]
        self.ffn = [FeedForward(em_dim, dff=512, name="ffn_{}".format(i)) for i in range(self.num_layers)]
        self.final = keras.layers.Dense(vocab_size, activation="softmax", name="out")
        self.adam_lr = extend_with_piecewise_linear_lr(Adam)

    def build(self):
        self.x_in = keras.Input(shape=(None,), name="inp")
        x = self.embedding(self.x_in)
        x_mask = self.mask(x)
        x = self.position_em(x)
        x = self.dropout_em(x, training=self.is_training)
        for i in range(self.num_layers):
            attn_out = self.mha[i]([x, x, x, x_mask])
            attn_out = self.dropout_attn[i](attn_out, training=self.is_training)
            x = keras.layers.Lambda(lambda x: x[0] + x[1], name="add_{}_1".format(i))([x, attn_out])
            x = keras.layers.LayerNormalization(epsilon=1e-6, name="LN_{}_1".format(i))(x)
            ffn_out = self.ffn[i](x)
            ffn_out = self.dropout_ffn[i](ffn_out, training=self.is_training)
            x = keras.layers.Lambda(lambda x: x[0] + x[1], name="add_{}_2".format(i))([x, ffn_out])
            x = keras.layers.LayerNormalization(epsilon=1e-6, name="LN_{}_2".format(i))(x)
        self.enc_out = self.final(x)
        self.model = keras.Model(self.x_in, self.enc_out)
        self.model.compile(loss=loss_function, metrics=[acc_function],
                           optimizer=self.adam_lr(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
```

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5qszvpibkSkzicib5IkvBWoItB4tfq55xL7blGbkO55507O6lHzwUibib5GUbC7qnN1RiaeEP8JOnqdJWrg/0?wx_fmt=png)

<center><font face="黑体" size=3>图13 Transformer 模型结构</font></center>

## 4 音频信号处理

在语音算法领域用的最多的音频特征是 MFCC（梅尔倒谱系数）和 Filter Bank（时频图，也称 fbank），两者整体相似只是 MFCC 比 fbank 多了一步 DCT （离散余弦变换）。

对语音信号分帧、加窗等一系列预处理后做短时傅里叶变换将其从时域转换到频域后得到的结果叫语谱图，经过 Mel 滤波后成为 mel 频谱，也叫 fbank，再做一层 DCT 就变成了 MFCC，这里我们声学模型的输入使用语谱图作为特征，可以借助 numpy 或者 librosa 实现。

```python
# 语谱图
spectrogram = np.abs(librosa.stft(wav_signal, n_fft=400, hop_length=160, 
                           win_length=400, window='hamming', center=False))
# mel频谱
mel_spec = librosa.feature.melspectrogram(wavsignal, sr=sr, n_fft=400, hop_length=160, 
                           win_length=400, window='hamming', center=False)
# mfcc
mfcc = librosa.feature.mfcc(wav_signal, sr=sr, n_fft=400, hop_length=160, 
                           win_length=400, window='hamming', center=False)
```

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5rsw1AoDj2Ykf7N8y79RCLiaib6lBO91icyP10ia8eDaHaTcCAib0WSIXI53zb1rZdAHF1Ivsg8JrxygNw/0?wx_fmt=png)

<center><font face="黑体" size=3>图14 语音信号各种频域特征之间的关系</font></center>

## 5 基于深度学习的中文语音识别实践

开发环境：`Python3.7.0+tensorflow2.1.0`

### 5.1 数据预处理

#### 5.1.1 建立标签映射关系

我们都知道，前面的模型中做的都是数学运算，但是我现在的数据都是文字和拼音，所以在模型运算前，我们需要把输入输入标签（拼音+声调、文字）转换成数字才能输入模型。既然需要转换，首先我们需要一个词典存储转换前的标签和转换后数字之间的映射关系，怎么获取字典？很简单，遍历一遍数据集给每个标签编个号即可。标签映射函数代码如下：

```python
def label2id(file_path, pny2id=None, han2id=None):
    if pny2id is None:
        pny2id = {"pad": 0, "unk": 1, "go": 2, "eos": 3, "": 4}
    if han2id is None:
        han2id = {"pad": 0, "unk": 1, "go": 2, "eos": 3, "": 4}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            # 数据集中语音文件名、拼音、汉字等3个字段是用制表符分割的
            wav_file, pnys, hans = line.strip().split('\t')
            if wav_file and pnys and hans:
                pny_lst = pnys.strip().split(" ")
                if len(pny_lst) == len(hans.replace(" ", "")):
                    for pny, han in zip(pny_lst, hans):
                        if pny not in pny2id:
                            pny2id[pny] = len(pny2id)
                        if han not in han2id:
                            han2id[han] = len(han2id)
    return pny2id, han2id
```

从函数中可以看到，映射字典中加入了5个特殊字符，`{"pad": 0, "unk": 1, "go": 2, "eos": 3, "": 4}` 分别表示填充标签、未登录标签、起始标签、终止标签、空标签。下面我们来生成两个字典，并将其写入文件中。

```python
import json

#生成字典
pny2id, han2id = label2id("./data_txt/aishell_dev.txt")
pny2id, han2id = label2id("./data_txt/aishell_train.txt", pny2id, han2id)
pny2id, han2id = label2id("./data_txt/aishell_test.txt", pny2id, han2id)
pny2id, han2id = label2id("./data_txt/prime.txt", pny2id, han2id)
pny2id, han2id = label2id("./data_txt/stcmd.txt", pny2id, han2id)
pny2id, han2id = label2id("./data_txt/thchs_dev.txt", pny2id, han2id)
pny2id, han2id = label2id("./data_txt/thchs_test.txt", pny2id, han2id)
pny2id, han2id = label2id("./data_txt/thchs_train.txt", pny2id, han2id)

# 写入文件
with open("./pny2id.json", "w", encoding="utf-8") as f:
    json.dump(pny2id, f)
with open("./han2id.json", "w", encoding="utf-8") as f:
    json.dump(han2id, f)
```

#### 5.1.2 构建DFCNN数据生成器

**计算CTC序列长度**：前文讲了，如果输出序列存在叠字的情况，那么 CTC 对应的序列会在叠字中间插入一个 $\epsilon$ 符号，插入之后 CTC 序列长度+1。

```python
def ctc_len(label):
    add_len = 0
    label_len = len(label)
    for i in range(label_len - 1):
        if label[i] == label[i + 1]:
            add_len += 1
    return label_len + add_len
```

**将一个 batch 内的音频特征数据 padding 成一个矩阵**：我们不能保证一个 batch 内所有的音频时间都一样长，那么通常提取出的音频特征矩阵肯定不是一样的大小，那怎么放进一个矩阵用来计算呢？找到最大的序列长度，建一个大矩阵（`batch_size, seq_len, depth`），把数据都往里面填，填不满的地方补零呗。

```python
def wav_padding(wav_data_lst):
    wav_lens = [data.shape[0] for data in wav_data_lst]
    wav_max_len = max(wav_lens)
    wav_lens = np.array([leng // 8 for leng in wav_lens])
    # 将每一个语音文件都padding到最大长度
    new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, wav_data_lst[0].shape[1]))
    for i in range(len(wav_data_lst)):
        new_wav_data_lst[i, :wav_data_lst[i].shape[0], :] = wav_data_lst[i]
    return new_wav_data_lst, wav_lens
```

**将一个 batch 内的标签数据 padding 成一个矩阵**：同样我们不能保证一个 batch 内所有的标签都一样长，那怎么放进一个矩阵用来计算呢？找到最大的长度，建一个大矩阵（`batch_size, seq_len`），把数据都往里面填，填不满的地方补零。

```python
def label_padding(label_data_lst):
    label_lens = [len(label) for label in label_data_lst]
    max_label_len = max(label_lens)
    new_label_data_lst = [label_data+[0]*(max_label_len - len(label_data)) for label_data in label_data_lst]
    return np.array(new_label_data_lst), np.array(label_lens)
```

万事俱备，咱们开始构建数据生成器：

```python
class AmDataGenerator:
    def __init__(self,
                 batch_size,
                 data_filter,
                 pny2id,
                 n_fft=400,
                 hop_length=160,
                 win_length=400,
                 window="hamming",
                 center=False):
        self.batch_size = batch_size
        self.data_filter = data_filter
        self.pny2id = pny2id
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center

    def get_am_batch(self):
        wav_data_lst = []
        label_data_lst = []
        while True:
            self.db = MongoClient(host="127.0.0.1", port=27017)["asr"]
            # for coll_name in self.db.list_collection_names():
            for coll_name in ["thchs30"]:
                for doc in self.db[coll_name].find(self.data_filter, batch_size=self.batch_size):
                    wav_str = doc["wav"]["str_data"]
                    wav_signal = np.frombuffer(wav_str, np.short).astype(np.float32)
                    sr = doc["wav"]["framerate"]
                    # string in list
                    pny = doc["pny"]
                    # string
                    han = doc["han"]
                    spectrogram = np.abs(librosa.stft(wav_signal,
                                                      n_fft=self.n_fft,
                                                      hop_length=self.hop_length,
                                                      win_length=self.win_length,
                                                      window=self.window,
                                                      center=self.center))
                    log_spec = np.log(spectrogram + 1).T
                    pad_log_spec = np.zeros((log_spec.shape[0] // 8 * 8 + 8, log_spec.shape[1]))
                    pad_log_spec[:log_spec.shape[0], :] = log_spec
                    label = [self.pny2id.get(p, 1) for p in pny]
                    label_ctc_len = ctc_len(label)
                    # 1、ctc最小的序列长度肯定不能小于模型输出帧数 2、显存有限，帧数大于800的数据咱就不要了哈
                    if pad_log_spec.shape[0] // 8 >= label_ctc_len and pad_log_spec.shape[0] <= 800:
                        wav_data_lst.append(pad_log_spec)
                        label_data_lst.append(label)
                    if len(wav_data_lst) == self.batch_size:
                        # 将wav_datapadding至最大长度，返回[batch_size, wav_max_len, half_window_len, 1)]
                        pad_wav_data, input_length = wav_padding(wav_data_lst)
                        pad_label_data, label_length = label_padding(label_data_lst)
                        yield [pad_wav_data, pad_label_data, input_length, label_length], np.zeros(
                            (pad_wav_data.shape[0],))
                        wav_data_lst, label_data_lst = [], []
```

#### 5.1.3 构建Transformer数据生成器

**将一个 batch 内的标签数据 padding 成一个矩阵**：找到最大的长度，建一个大矩阵（`batch_size, seq_len`），把数据都往里面填，填不满的地方补零。这里和前面 `label_padding` 函数几乎一样，只是不返回 `label_length`。

```python
def seq_padding(x):
    length = [len(item) for item in x]
    max_len = max(length)
    x = [item+[0]*(max_len-len(item)) for item in x]
    return np.array(x)
```

构建Transformer数据生成器。

```python
class LmDataGenerator:
    def __init__(self, batch_size, data_filter, pny_vocab, han_vocab):
        self.data_filter = data_filter
        self.batch_size = batch_size
        self.pny2id = pny_vocab
        self.han2id = han_vocab

    def get_lm_batch(self):
        pny_lst = []
        han_lst = []
        while True:
            self.db = MongoClient(host="127.0.0.1", port=27017)["asr"]
            # for coll_name in self.db.list_collection_names():
            for coll_name in ["thchs30"]:
                for doc in self.db[coll_name].find(self.data_filter, batch_size=self.batch_size):
                    pny = doc["pny"]
                    han = doc["han"]
                    pny = [self.pny2id.get(p, 1) for p in pny]
                    han = [self.han2id.get(h, 1) for h in han]
                    if len(pny) <= 512 and len(pny) == len(han):
                        pny_lst.append(pny)
                        han_lst.append(han)
                    if len(pny_lst) == self.batch_size:
                        inp = seq_padding(pny_lst)
                        label = seq_padding(han_lst)
                        yield inp, label
                        pny_lst, han_lst = [], []
```

### 5.2 开始训练DFCNN

#### 5.2.1 导入库并设置GPU显存动态增长

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import json
from pymongo import MongoClient
from network import DFCNN, Evaluate
from utils import DataGenerator
import os
import datetime
%load_ext tensorboard


gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

#### 5.2.2 划分训练集

`id` 能被10整除的作为验证集，不能整除的作为训练集

```python
train_data_filter = {"id":{"$not": {"$mod":[10, 0]}}}
val_data_filter = {"id":{"$mod":[10, 0]}}
```

#### 5.2.3 读取标签映射关系

```python
with open("pny2id.json", "r", encoding="utf-8") as f:
    pny2id = json.load(f)
pny2id["blank"] = len(pny2id)
id2pny = {v:k for k,v in pny2id.items()}
print(len(pny2id), pny2id["blank"])

# output:
1300 1299
```

#### 5.2.4 获取train_batch

```python
# 笔者电脑垃圾，有条件可以设置大一点
batch_size = 16
train_data_gen = AmDataGenerator(batch_size, train_data_filter, pny2id)
train_batch = train_data_gen.get_am_batch()
```

#### 5.2.5 实例化训练模型

```python
dfcnn = DFCNN(vocab_size=len(pny2id), inp_width=201, is_training=True)
dfcnn.model.summary()
train_model = dfcnn.ctc_model

# output:
Model: "model"
_____________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to         
=====================================================================================
the_inputs (InputLayer)         [(None, None, 201)]  0                               
masking (Masking)               (None, None, 201)    0           the_inputs[0][0]     
expand_dims (Lambda)            (None, None, 201, 1) 0           masking[0][0]       
init_mask (Lambda)              (None, None)         0           masking[0][0]       
cnn_cell (CNNCell)              ((None, None, 100, 3 9824        expand_dims[0][0]   
cnn_cell_1 (CNNCell)            ((None, None, 50, 64 55936       cnn_cell[0][0]       
cnn_cell_2 (CNNCell)            ((None, None, 25, 12 222464      cnn_cell_1[0][0]     
cnn_cell_3 (CNNCell)            ((None, None, 25, 12 296192      cnn_cell_2[0][0]     
reshape (Reshape)               (None, None, 3200)   0           cnn_cell_3[0][0]     
dense_1 (Dense)                 (None, None, 256)    819456      dropout[0][0]       
dropout_1 (Dropout)             (None, None, 256)    0           dense_1[0][0]       
dense_2 (Dense)                 (None, None, 1300)   334100      dropout_1[0][0]     
lambda (Lambda)                 (None, None, 1300)   0           dense_2[0][0]       
                                                                 cnn_cell_3[0][1]     
=====================================================================================
Total params: 1,737,972
Trainable params: 1,736,564
Non-trainable params: 1,408
```

可见，模型共有约173万个参数需要训练。

#### 5.2.6 设置回调

`callback` 有很多种，笔者这里就简单弄两种：`Evaluate` 和 `Temsorboard`。

```python
class AmEvaluate(keras.callbacks.Callback):
    def __init__(self, val_filter, batch_size, pny2id, **kwargs):
        super(AmEvaluate, self).__init__(**kwargs)
        self.val_filter = val_filter
        self.batch_size = batch_size
        self.pny2id = pny2id
        self.min_loss = 1000.0

    def on_epoch_end(self, epoch, logs=None):
        val_batch = AmDataGenerator(self.batch_size, self.val_filter, self.pny2id).get_am_batch()
        loss = self.model.evaluate_generator(val_batch, steps=1000)
        if loss < self.min_loss:
            self.min_loss = loss
            print("   model improved! val_loss: {}".format(loss))
            self.model.save_weights("model_weights/cnn_ctc.weights")
        else:
            print("   current loss: {}".format(loss))


am_evaluate = AmEvaluate(val_data_filter, batch_size, pny2id)
log_dir = os.path.join(".\\logs\\fit" , datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
%tensorboard --logdir logs/fit
```

#### 5.2.7 训练模型

训练100轮，每轮800个 `batch` ，这里 `fit_generator` 函数好像要被移除了，新版本直接使用 `fit` 函数也可以训练生成器。

```python
epochs = 100
train_model.fit_generator(train_batch, steps_per_epoch=800, epochs=epochs,
                          callbacks=[am_evaluate, tensorboard_callback]
                         )

#output:
Train for 800 steps
Epoch 1/100
799/800 [============================>.] - ETA: 0s - loss: 65.0819
··· 此处省略一万行
Epoch 100/100
799/800 [============================>.] - ETA: 0s - loss: 18.3634
   current loss: 82.56352760314941
800/800 [==============================] - 346s 432ms/step - loss: 18.3643
```

#### 5.2.8 评估模型

先定义解码器，对模型输出进行 `ctc 解码`，将其转化成对应的标签。

```python
# 定义解码器
def decode_ctc(num_result, num2word):
    result = num_result[:, :, :]
    in_len = np.zeros((1), dtype=np.int32)
    in_len[0] = result.shape[1]
    r = keras.backend.ctc_decode(result, in_len, greedy=True, beam_width=10, top_paths=1)
    r1 = keras.backend.get_value(r[0][0])
    r1 = r1[0]
    text = []
    for i in r1:
        text.append(num2word[i])
    return r1, text
```

简单看下效果：

```python
for i in range(5):
    print('\n the ', i, 'th example.')
    # 载入训练好的模型，并进行识别
    ctc_inputs, ctc_out = next(test_batch)
    result = dfcnn.model.predict(ctc_inputs[0])
    # 将数字结果转化为文本结果
    _, text = decode_ctc(result, id2pny)
    pny = [id2pny.get(int(i), "*") for i in ctc_inputs[1][0]]
    print('识别结果：', " ".join(text))
    print('原文结果：', " ".join(pny))
   
#output:
the  0 th example.
识别结果： ke3 shei2 zhi1 reng2 wan2 hou4 ta1 yi1 zhao4 jin4 zi zhe4 jian4 zuo3 xia4 yan2 ye3 de xian4 you4 cu1 you3 hei1 yi2 you4 cuo4 yu3 xian3 bu2 dui4 cheng1
原文结果： ke3 shei2 zhi1 wen2 wan2 hou4 ta1 yi1 zhao4 jing4 zi zhi1 jian4 zuo3 xia4 yan3 jian3 de xian4 you4 cu1 you4 hei1 yu3 you4 ce4 ming2 xian3 bu2 dui4 cheng1

 the  1 th example.
识别结果： er4 ye4 si4 ri4 zhe4 jin4 xin1 xi1 men2 wai4 luo2 jia1 nian3 wang2 jia1 gang1 zhe4 ju4 qing1 wen2 shi4 te4 di4 cong2 dong1 men2 wai4 gan3 lai2 ci4 pao4
原文结果： er4 yue4 si4 ri4 zhu4 jin4 xin1 xi1 men2 wai4 luo2 jia1 nian3 wang2 jia1 gang1 zhu1 zi4 qing1 wen2 xun4 te4 di4 cong2 dong1 men2 wai4 gan3 lai2 qing4 he4

 the  2 th example.
识别结果： dou1 yong4 cao3 mao4 huo4 ge1 zhang3 hu4 kuo4 jiu3 wai4 da3 lie4 qie ju1 chuan1 guo4 mai4 ni2 tang2 bang1 de yuan4 ba4 kao4 hui2 zi4 ji3 de su4 she4 qu4 le
原文结果： dou1 yong4 cao3 mao4 huo4 ge1 bo zhou3 hu4 zhe wan3 lie4 lie4 qie ju1 chuan1 guo4 lan4 ni2 tang2 ban1 de yuan4 ba4 pao3 hui2 zi4 ji3 de su4 she4 qu4 le

 the  3 th example.
识别结果： jun1 zi duo1 yu4 ze2 kang1 mu4 fu4 gui4 wang3 dao4 su4 huo4 xiang4 ren2 duo1 yu4 ze2 duo1 qiu2 bao3 yong4 bai4 jia1 sang4 shen1 pad
原文结果： jun1 zi duo1 yu4 ze2 tan1 mu4 fu4 gui4 wang3 dao4 su4 huo4 xiao3 ren2 duo1 yu4 ze2 duo1 qiu2 wang4 yong4 bai4 jia1 sang4 shen1

 the  4 th example.
识别结果： wo3 men jia1 you3 tai2 jiu4 yin1 xiang3 huai4 le yi3 hou4 wo4 ma1 rang4 wo3 xiu1 bei4 wo3 yue4 xiu1 yue4 bu4 cheng2 yang4 zi ta1 jiu4 guo4 wang3 huo2 yao4 le pad
原文结果： wo3 men jia1 you3 tai2 jiu4 yin1 xiang3 huai4 le yi3 hou4 wo3 ma1 rang4 wo3 xiu1 bei4 wo3 yue4 xiu1 yue4 bu4 cheng2 yang4 zi ta1 jiu4 bu4 wang3 hui2 yao4 le
```

效果还行吧。

### 5.3 开始训练Transformer

#### 5.3.1 导入库并设置显存自动增长

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import json
from pymongo import MongoClient
from network import DFCNN, Evaluate
from utils import AmDataGenerator
import os
import datetime
%load_ext tensorboard


gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

#### 5.3.2 划分训练集

`id` 能被10整除的作为验证集，不能整除的作为训练集

```python
train_data_filter = {"id":{"$not": {"$mod":[10, 0]}}}
val_data_filter = {"id":{"$mod":[10, 0]}}
```

#### 5.3.3 读取标签映射关系

```python
with open("pny2id.json", "r", encoding="utf-8") as f:
    pny2id = json.load(f)
id2pny = {v:k for k,v in pny2id.items()}
with open("han2id.json", "r", encoding="utf-8") as f:
    han2id = json.load(f)
id2han = {v:k for k,v in han2id.items()}
```

#### 5.3.4 获取train_batch

```python
batch_size = 64
em_dim = 128
train_batch = LmDataGenerator(batch_size, train_data_filter, pny2id, han2id).get_lm_batch()
```

#### 5.3.5 实例化训练模型

```python
lm = Transformer(len(han2id), em_dim, 2, True)
lm.build()
lm.model.summary()

#output:
Model: "model"
_____________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to         
=====================================================================================
inp (InputLayer)                [(None, None)]       0                               
embedding (Embedding)           (None, None, 128)    805632      inp[0][0]           
position (PositionalEncoding)   (None, None, 128)    0           embedding[0][0]     
dropout_em (Dropout)            (None, None, 128)    0           position[0][0]       
mask (Lambda)                   (None, None)         0           embedding[0][0]     
multiheadAttn_0 (MultiHeadAtten (None, None, 128)    49152       dropout_em[0][0]     
                                                                 dropout_em[0][0]     
                                                                 dropout_em[0][0]     
                                                                 mask[0][0]           
enc_0_dropout (Dropout)         (None, None, 128)    0           multiheadAttn_0[0][0]            
add_0_1 (Lambda)                (None, None, 128)    0           dropout_em[0][0]     
                                                                 enc_0_dropout[0][0] 
LN_0_1 (LayerNormalization)     (None, None, 128)    256         add_0_1[0][0]       
ffn_0 (FeedForward)             (None, None, 128)    131712      LN_0_1[0][0]         
ffn_0_dropout (Dropout)         (None, None, 128)    0           ffn_0[0][0]         
add_0_2 (Lambda)                (None, None, 128)    0           LN_0_1[0][0]         
                                                                 ffn_0_dropout[0][0] 
LN_0_2 (LayerNormalization)     (None, None, 128)    256         add_0_2[0][0]       
multiheadAttn_1 (MultiHeadAtten (None, None, 128)    49152       LN_0_2[0][0]         
                                                                 LN_0_2[0][0]         
                                                                 LN_0_2[0][0]         
                                                                 mask[0][0]           
enc_1_dropout (Dropout)         (None, None, 128)    0           multiheadAttn_1[0][0]            
add_1_1 (Lambda)                (None, None, 128)    0           LN_0_2[0][0]         
                                                                 enc_1_dropout[0][0] 
LN_1_1 (LayerNormalization)     (None, None, 128)    256         add_1_1[0][0]       
ffn_1 (FeedForward)             (None, None, 128)    131712      LN_1_1[0][0]         
ffn_1_dropout (Dropout)         (None, None, 128)    0           ffn_1[0][0]         
add_1_2 (Lambda)                (None, None, 128)    0           LN_1_1[0][0]         
                                                                 ffn_1_dropout[0][0] 
LN_1_2 (LayerNormalization)     (None, None, 128)    256         add_1_2[0][0]       
out (Dense)                     (None, None, 6294)   814893      LN_1_2[0][0]         
=====================================================================================
Total params: 1,986,221
Trainable params: 1,986,221
Non-trainable params: 0
```

可见，模型共有约198万个参数需要训练。

#### 5.3.6 设置回调

这里`callback` 同样简单弄两种：`Evaluate` 和 `Temsorboard`。

```python
class LmEvaluate(keras.callbacks.Callback):
    def __init__(self, val_filter, batch_size, pny2id, han2id, **kwargs):
        super(LmEvaluate, self).__init__(**kwargs)
        self.val_filter = val_filter
        self.bacth_size = batch_size
        self.pny2id = pny2id
        self.han2id = han2id
        self.max_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_batch = LmDataGenerator(self.bacth_size, self.val_filter, self.pny2id, self.han2id).get_lm_batch()
        loss, acc = self.model.evaluate_generator(val_batch, steps=300)
        if acc >= self.max_acc:
            print("    model improved!  val_loss:{} val_acc:{}".format(loss, acc))
            self.max_acc = acc
            self.model.save_weights("./model_weights/transformer.weights")
        else:
            print("    current val_loss:{} val_acc:{}".format(loss, acc))
            

lm_evaluate = LmEvaluate(val_data_filter, batch_size, pny2id, han2id)
log_dir = os.path.join(".\\logs\\lmfit" , datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
%tensorboard --logdir logs/lmfit
```

#### 5.3.7 训练模型

训练100轮，每轮800个 `batch` 。

```python
epochs = 100
lm.model.fit_generator(train_batch, steps_per_epoch=800, epochs=epochs,
                       callbacks=[evaluate],
                      )

#output:
Epoch 1/100
799/800 [============================>.] - ETA: 0s - loss: 0.4780 - acc_function: 0.8422    model improved!  val_loss:0.5599649414420128 val_acc:0.8189043402671814
800/800 [==============================] - 3133s 209ms/step - loss: 0.4780 - acc_function: 0.8422
Epoch 2/100
799/800 [===========>..................] - ETA: 27:30 - loss: 0.4853 - acc_function: 0.8402
```

#### 5.3.8 评估模型

简单看下效果：

```python
for i in range(5):
    print('\n the ', i, 'th example.')
    # 载入训练好的模型，并进行识别
    inp, label = next(test_batch)
    preds = lm.model.predict(inp)
    got = ''.join(id2han[idx] for idx in np.argmax(preds[0], axis=-1))
    han = ''.join(id2han[idx] for idx in label[0])
    print('原文汉字：', han)
    print('识别结果：', got)

#outputs:
the  0 th example.
原文汉字： 二毛你今天沒课嘛还和李霞聊天
识别结果： 二毛你今天没客吗还和理辖聊天

 the  1 th example.
原文汉字： 波涛汹涌的好多傻说半天
识别结果： 波涛兄涌的好多傻说半天

 the  2 th example.
原文汉字： 好好准备准备几个台词
识别结果： 好好准备准备几个台词

 the  3 th example.
原文汉字： 让你失望了我们开卷佳儿
识别结果： 让你失望了我们开卷家而

 the  4 th example.
原文汉字： 那卫媛是真的怀孕吗
识别结果： 那为院是真的怀孕吗
```

效果还行吧，准确率大概81%，有时间再调调。

### 5.4 模型预测

#### 5.4.1 数据生成器

声学模型和语言模型都训练完毕就可以进行语音识别了。简单写一个预测要使用的数据生成器，把音频特征、拼音、文字都返回。

```python
class ASRDataGenerator(object):
    def __init__(self,
                 batch_size,
                 data_filter,
                 pny2id,
                 mongo_coll_names,
                 n_fft=400,
                 hop_length=160,
                 win_length=400,
                 window="hamming",
                 center=False):
        self.batch_size = batch_size
        self.data_filter = data_filter
        self.pny2id = pny2id
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.coll_names = mongo_coll_names

    def get_asr_batch(self):
        wav_data_lst = []
        while True:
            self.db = MongoClient(host="127.0.0.1", port=27017)["asr"]
            # for coll_name in self.db.list_collection_names():
            for coll_name in self.coll_names:
                for doc in self.db[coll_name].find(self.data_filter, batch_size=self.batch_size):
                    wav_str = doc["wav"]["str_data"]
                    wav_signal = np.frombuffer(wav_str, np.short).astype(np.float32)
                    sr = doc["wav"]["framerate"]
                    # string in list
                    pny = doc["pny"]
                    # string
                    han = doc["han"]
                    spectrogram = np.abs(librosa.stft(wav_signal,
                                                      n_fft=self.n_fft,
                                                      hop_length=self.hop_length,
                                                      win_length=self.win_length,
                                                      window=self.window,
                                                      center=self.center))
                    log_spec = np.log(spectrogram + 1).T
                    pad_log_spec = np.zeros((log_spec.shape[0] // 8 * 8 + 8, log_spec.shape[1]))
                    pad_log_spec[:log_spec.shape[0], :] = log_spec
                    label = [self.pny2id.get(p, 1) for p in pny]
                    label_ctc_len = ctc_len(label)
                    # 1、ctc最小的序列长度肯定不能小于模型输出帧数 2、显存有限，帧数大于800的数据咱就不要了哈
                    if pad_log_spec.shape[0] // 8 >= label_ctc_len and pad_log_spec.shape[0] <= 800:
                        wav_data_lst.append(pad_log_spec)
                    if len(wav_data_lst) == self.batch_size:
                        # 将wav_datapadding至最大长度，返回[batch_size, wav_max_len, half_window_len, 1)]
                        pad_wav_data, input_length = wav_padding(wav_data_lst)
                        yield pad_wav_data, pny, han
                        wav_data_lst = []
```

#### 5.4.2 加载模型

实例化声学模型和语言模型并加载权重:

```python
dfcnn = DFCNN(vocab_size=len(pny2id)+1, inp_width=201, is_training=False)
dfcnn.model.load_weights("./model_weights/epoch_100_800_dfcnn_202201040702.weights")

transformer = Transformer(len(han2id), em_dim=128, num_layers=2, is_training=False)
transformer.build()
transformer.model.load_weights("./model_weights/epoch_200_800_transformer_202201041552.weights")
```

#### 5.4.3 编辑距离

计算编辑距离作为错误指标：

```python
# 编辑距离
def GetEditDistance(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':
            leven_cost += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            leven_cost += (j2 - j1)
        elif tag == 'delete':
            leven_cost += (i2 - i1)
    return leven_cost
```

#### 5.4.4 简单评测

话不多说，看下效果吧。

```python
word_num = 0
word_error_num = 0
for i in range(5):
    print('\n the ', i, 'th example.')
    # 载入训练好的模型，并进行识别
    inputs, pny, han = next(asr_batch)
    result = dfcnn.model.predict(inputs)
    # 将数字结果转化为文本结果
    _, text = decode_ctc(result, id2pny)
    text_str = ' '.join(text)
    print('原文拼音：', pny)
    print('识别拼音：', text_str)
    
    x = np.array([pny2id.get(pny, 1) for pny in text])
    x = x.reshape(1, -1)
    preds = transformer.model.predict(x)
    got = ''.join(id2han[idx] for idx in np.argmax(preds[0], axis=-1))
    print('原文汉字：', han)
    print('识别文字：', got)
    word_error_num += min(len(han), GetEditDistance(han, got))
    word_num += len(han)
word_error_num / word_num

#outputs:
the  0 th example.
原文拼音： zhong1 guo2 chu2 ge4 bie2 yun4 dong4 yuan2 yu3 wai4 guo2 xuan3 shou3 zu3 dui4 can1 jia1 shuang1 da3 yu4 xuan3 sai4 wai4 jun1 jiang1 zhi2 jie1 jin4 ru4 zheng4 sai4
识别拼音： zhong1 guo2 chu2 ge4 bie2 yun4 dong4 yuan2 yu3 wai4 guo2 xuan3 shou3 zu3 dui4 can1 jia1 shuang1 da3 yu4 xuan3 sai4 wai4 jun1 jiang1 zhi2 jie1 jin4 ru4 zheng4 sai4
原文汉字： 中国除个别运动员与外国选手组对参加双打预选赛外均将直接进入正赛
识别文字： 中国除个别运动员与外国选手组对参家双打预选赛外军将直接进入正赛

the  1 th example.
原文拼音： dan4 bu2 yao4 dui4 ma3 dao3 yi1 yong1 er2 shang4 luan4 chao3 luan4 shuo1 fang2 zhi3 ji1 hua4 mao2 dun4 fa1 sheng1 yi4 wai4 chong1 tu1
识别拼音： dan4 bu2 yao4 dui4 ma3 dao3 yi1 yong1 er2 shang4 luan4 chao3 luan4 shuo1 fang2 zhi3 ji1 hua4 mao2 dun4 fa1 sheng1 yi4 wai4 chong1 tu1
原文汉字： 但不要对马导一拥而上乱吵乱说防止激化矛盾发生意外冲突
识别文字： 但不要对马导一拥而上乱炒乱说房只机化毛盾发生意外充突

the  2 th example.
原文拼音： niao3 niao3 de ta1 chui1 wo3 dao4 chen2 si3 de ye4 bang1 wo3 wang4 an1 jing4 de ling2 hun2 men zai4
识别拼音： niao3 niao3 de ta1 chui1 wo3 dao4 chen2 si3 de ye4 bang1 wo3 wang4 an1 jing4 de ling2 hun2 men zai4
原文汉字： 袅袅地他吹我到沉死的夜邦我望安静的灵魂们在
识别文字： 鸟鸟的他吹我到陈死的业帮我望安境的零魂们在

the  3 th example.
原文拼音： ci3 xiao1 xi1 yi1 chu1 li4 ji2 hong1 dong4 le quan2 yuan4 yi1 xie1 qin1 peng2 gu4 you3 fen1 fen1 chu1 mian4 quan4 zu3 gao1 shu4 huai2 yao4 san1 si1 er2 hou4 xing2
识别拼音： ci3 xiao1 xi1 yi1 chu1 li4 ji2 hong1 dong4 le quan2 yuan4 yi1 xie1 qin1 peng2 gu4 you3 fen1 fen1 chu1 mian4 quan4 zu3 gao1 shu4 huai2 yao4 san1 si1 er2 hou4 xing2
原文汉字： 此消息一出立即轰动了全院一些亲朋故友纷纷出面劝阻高树槐要三思而后行
识别文字： 此消西一出利及轰动了全院一些亲朋故有分分出面劝组高术怀要三斯而后行

the  4 th example.
原文拼音： ni3 men da4 jia1 chi1 cang1 ying2 wo3 ye3 gen1 zhe chi1 cang1 ying2 ni3 men yuan4 yi4 chi1 wen2 zi wo3 jiu4 gen1 zhe chi1 wen2 zi shen2 me shi4 bu4 yong4 wen4 wo3
识别拼音： ni3 men da4 jia1 chi1 cang1 ying2 wo3 ye3 gen1 zhe chi1 cang1 ying2 ni3 men yuan4 yi4 chi1 wen2 zi wo3 jiu4 gen1 chi1 wen2 zi shen2 me shi4 bu4 yong4 wen4 wo3 wang3
原文汉字： 你们大家吃苍蝇我也跟着吃苍蝇你们愿意吃蚊子我就跟着吃蚊子什么事不用问我
识别文字： 你们大家吃仓营我也跟着吃苍营你们院意吃文子我就跟吃文子什么是不用问我网

0.23972602739726026
```

错误率大概在 `0.23972602739726026`，效果还可以，原本以为硬train一发效果完全不能看，结果准确率还行，调调参数应该会有提高。

## 6 小结

花了3周时间终于写了这篇一万三千字的博客，年底实在太忙了QAQ，本文主要介绍了一个完整的基于深度学习的端到端语音识别过程。从数据集获取、数据库存取，到模型结构拆分讲解，再到音频特征的提取，最后训练模型并进行预测，每个环节都配有完整的代码和代码讲解，对语音识别初学者具有很好的引导作用。
