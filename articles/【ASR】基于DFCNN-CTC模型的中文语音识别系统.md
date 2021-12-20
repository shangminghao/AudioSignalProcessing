**写在前面：**时间是让人猝不及防的东西，恍惚间半年就过去了，笔者这半年工作太忙，学习近乎停滞，回想起来懊悔不已，半年时间浪费了，从这个月开始要振作起来，无论多忙都要保持一颗求学上进的心。及时当勉励，岁月不待人！

## 1 背景介绍

笔者前段时间停车充电的时候看了李宏毅的NLP课程，讲的很生动，想起来之前笔者也学习过语音识别的一些浅显知识，觉得有必要总结一下，于是有了这篇文章，本文主要介绍模型结构和具体实现细节，关于 ASR（Automatic Speech Recognition）的发展背景等情况不再赘述。

本文搭建了一个完整的中文语音识别系统，包括声学模型和语言模型，能够将输入的音频信号识别为汉字。本文主要从以下四个方面进行介绍：**数据集、模型结构、音频信号处理、实践环节**。

完整的中文语音识别系统包括两个模型：将语音信号转换为“拼音+声调”的**声学模型**，将“拼音+声调”转换为中文文字的**语言模型**。为什么用两个模型？直接一个模型从语音信号到文字，硬train一发不好吗？如果只是一个孤立词识别的语音识别系统，我相信硬train一发应该也能效果不错，但是对于句子级的语音识别任务，除了单纯地识别出发音元素以外，还需要识别出语音的含义转化为一串中文文字，这个过程复杂度更高，一个模型效果通常不太好。所以通常我们需要声学模型将声学和发音学的知识进行整合，将语音信号经过特征处理后作为声学模型的输入，经过转换后输出可变长的发音元素，中文来讲就是“拼音+声调”。然后我们需要一个语言模型学习“拼音+声调”与文字之间的相互关系，将“拼音+声调”作为语音模型的输入，输出中文文字。本文声学模型使用科大讯飞提出的DFCNN深度全序列卷积神经网络，语言模型则使用transformer模型搭建拼音序列生成汉字序列系统。

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



