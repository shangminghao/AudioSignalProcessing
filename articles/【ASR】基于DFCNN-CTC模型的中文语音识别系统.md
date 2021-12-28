**写在前面：**时间是让人猝不及防的东西，恍惚间半年就过去了，笔者这半年工作太忙，学习近乎停滞，回想起来懊悔不已，半年时间浪费了，从这个月开始要振作起来，无论多忙都要保持一颗求学上进的心。及时当勉励，岁月不待人！

## 1 背景介绍

笔者前段时间停车充电的时候看了李宏毅的NLP课程，讲的很生动，想起来之前笔者也学习过语音识别的一些浅显知识，觉得有必要总结一下，于是有了这篇文章，本文主要介绍模型结构和具体实现细节，关于 ASR（Automatic Speech Recognition）的发展背景等情况不再赘述。

本文搭建了一个完整的中文语音识别系统，包括声学模型和语言模型，能够将输入的音频信号识别为汉字。本文主要从以下四个方面进行介绍：**数据集、模型结构、音频信号处理、实践环节**。

完整的中文语音识别系统包括两个模型：将语音信号转换为“拼音+声调”的**声学模型**，将“拼音+声调”转换为中文文字的**语言模型**。为什么用两个模型？直接一个模型从语音信号到文字，硬train一发不好吗？如果只是一个孤立词识别的语音识别系统，我相信硬train一发应该也能效果不错，但是对于句子级的语音识别任务，除了单纯地识别出发音元素以外，还需要识别出语音的含义转化为一串中文文字，这个过程复杂度更高，一个模型效果通常不太好。所以通常我们需要声学模型将声学和发音学的知识进行整合，将语音信号经过特征处理后作为声学模型的输入，经过转换后输出可变长的发音元素，中文来讲就是“拼音+声调”。然后我们需要一个语言模型学习“拼音+声调”与文字之间的相互关系，将“拼音+声调”作为语音模型的输入，输出中文文字。本文声学模型使用科大讯飞提出的DFCNN深度全序列卷积神经网络，语言模型则使用transformer模型搭建拼音序列生成汉字序列系统。

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5qH9tenE7VPa6cIYCIzGJIYgyOiao2q7j6PuqVzkOEDhmARhQQNlwO17IbLFj24ZpibKDl8KA1Ceo1g/0?wx_fmt=png)

<center><font face="黑体" size=3>图1 声学模型的任务</font></center>

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5qH9tenE7VPa6cIYCIzGJIYfYj0HO6a4qPH0Pkf1y84wDv4sfibW4LwJF0icXV8jqxC5U5icpH10NL6Q/0?wx_fmt=png)

<center><font face="黑体" size=3>图2 语言模型的任务</font></center>

## 2 数据集

本文数据集采用了 [OpenSLR](http://www.openslr.org/resources.php) 上的中文免费数据，包括：thchs-30、aishell、primewords、st-cmd四个数据集，训练集总计大约450个小时。数据标签整理在`data`路径下，包含了解压后数据的路径，以及训练所需的数据标注格式，其中primewords、st-cmd目前未区分训练集和测试集，需要手工区分，各个数据集的规模如下：

|    Name    | train  |  dev  | test |
| :--------: | :----: | :---: | :--: |
|  aishell   | 120098 | 14326 | 7176 |
| primewords | 40783  | 5046  | 5073 |
|  thchs-30  | 10000  |  893  | 2495 |
|   st-cmd   | 10000  |  600  | 2000 |

数据集总共约`40G`，无法全部加载进内存，若个 epoch 都去磁盘里面执行 wave 读取操作，速度较慢。这里我将所有语音文件和对应的数据标签全部存入 MongoDB 数据库，需要的时候再去调用。

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

由图4可见，DFCNN 模型输入是一个声音信号的语谱图（Spectrogram），那么什么是语谱图，语谱图怎么求？这点我们在下节介绍。输入层紧接着的是N个由卷积层、归一化层、池化层组成的卷积神经网络结构，也是模型的主要组成单元，这里我们称为**卷积单元**。N个卷积单元之后接了一个 `Reshape` 层，这一层可以将上一层的输出由4维转换为3维，即从 `[batch_size, seq_len, width, depth]` 转换为 `[batch_size, seq_len, width* depth]` 然后就是两层全连接神经网络，模型训练时中间夹了 `Dropout` 用于缓解过拟合问题。

#### 3.1.1 卷积单元

**卷积层：**每个卷积单元里包含两个卷积层，卷积核大小都为 `3*3` ，padding 方式为 `same` ，激活函数设置为 `relu` ，卷积核数量为32、64、128等。

**归一化层：**这里使用 `BatchNormlization`，在一个 `batch` 内从 `length, width, depth` 三个方向对输入向量进行标准化，这样做可以使神经网络每一层的输入基本分布在一个标准差和均值下，**目的就是让每一层的分布稳定下来**，让后面的层可以在前面层的基础上安心学习知识，解决深层网络下的梯度消失问题。

**池化层：**使用 `Maxpooling` 方式，池化尺寸为 `2*2` ，步长和池化尺寸一致，padding方式为 `valid` ，所以池化之后尺寸减半。这一层作用是减小输出向量尺寸大小和缓解过拟合。

```python
class CNNCell(keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3,3), activation='relu', pool=True, **kwargs):
        super(CNNCell, self).__init__(**kwargs)
        self.conv2d_1 = keras.layers.Conv2D(filters, 
                                            kernel_size, 
                                            use_bias=True,
                                            activation=activation,
                                            padding='same',
                                            kernel_initializer='he_normal',)
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

    def call(self, inputs, **kwargs):
        x = self.conv2d_1(inputs)
        x = self.batch_norm_1(x)
        x = self.conv2d_2(x)
        x = self.batch_norm_2(x)
        if self.pool:
            x = self.max_pool(x)
        return x
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] // 2, input_shape[2] // 2, input_shape[3] // 2
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
        self._model_init()
        if self.is_training:
            self._ctc_init()
            self.opt_init()

    def _model_init(self):
        self.inputs = keras.layers.Input(name='the_inputs', shape=(None, self.inp_width, 1))
        x = CNNCell(32)(self.inputs)
        x = CNNCell(64)(x)
        x = CNNCell(128)(x)
        x = CNNCell(128, pool=False)(x)
        x = keras.layers.Reshape((-1, self.inp_width // 8 * 128), name="reshape")(x)
        x = keras.layers.Dropout(rate=0.2)(x, training=self.is_training)
        x = keras.layers.Dense(256, activation="relu",
                               use_bias=True, kernel_initializer='he_normal')(x)
        x = keras.layers.Dropout(rate=0.2)(x, training=self.is_training)
        self.outputs = keras.layers.Dense(self.vocab_size, activation='softmax',
                               use_bias=True, kernel_initializer='he_normal')(x)
        self.model = keras.Model(inputs=self.inputs, outputs=self.outputs)

    def _ctc_init(self):
        self.labels = keras.Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = keras.Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = keras.Input(name='label_length', shape=[1], dtype='int64')
        self.loss_out = self.ctc_batch_cost_layer([self.labels, self.outputs, self.input_length, self.label_length])
        self.ctc_model = keras.Model(inputs=[self.inputs, self.labels, self.input_length, self.label_length],
                                     outputs=self.loss_out, name="_ctc_model")

    def opt_init(self):
        opt = keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.01, epsilon=10e-8)
        # if self.gpu_nums > 1:
        #     self.ctc_model = keras.utils.multi_gpu_model(self.ctc_model,gpus=self.gpu_nums)
        # keras自定义损失函数要求函数前两个参数分别为y_true, y_pred
        self.ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt)
```

![](https://mmbiz.qpic.cn/mmbiz_png/GJUG0H1sS5qH9tenE7VPa6cIYCIzGJIY4mn82caNZAJUFlu5VLzd8zYZ7cIXHegIV5ubbq6X4xkic2U1VpThiaGw/0?wx_fmt=png)

### 3.2 语言模型

前面讲过语言模型的任务是将发音单元转换为文字，具体就是把“拼音+声调”转换为一串文字，
