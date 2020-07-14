# Task4 基于深度学习的文本分类1

在上一章节，我们使用传统机器学习算法来解决了文本分类问题，从本章开始我们将尝试使用深度学习方法。

## 基于深度学习的文本分类

与传统机器学习不同，深度学习既提供特征提取功能，也可以完成分类的功能。从本章开始我们将学习如何使用深度学习来完成文本表示。

### 学习目标

- 学习FastText的使用和基础原理
- 学会使用验证集进行调参

### 文本表示方法 Part2

#### 现有文本表示方法的缺陷

在上一章节，我们介绍几种文本表示方法：

- One-hot
- Bag of Words
- N-gram
- TF-IDF

也通过sklean进行了相应的实践，相信你也有了初步的认知。但上述方法都或多或少存在一定的问题：转换得到的向量维度很高，需要较长的训练实践；没有考虑单词与单词之间的关系，只是进行了统计。

与这些表示方法不同，深度学习也可以用于文本表示，还可以将其映射到一个低纬空间。其中比较典型的例子有：FastText、Word2Vec和Bert。在本章我们将介绍FastText，将在后面的内容介绍Word2Vec和Bert。

#### FastText

FastText是一种典型的深度学习词向量的表示方法，它非常简单通过Embedding层将单词映射到稠密空间，然后将句子中所有的单词在Embedding空间中进行平均，进而完成分类操作。

所以FastText是一个三层的神经网络，输入层、隐含层和输出层。

<img src="img/fast_text.png" alt="fasttext" style="zoom:50%;" />

下图是使用keras实现的FastText网络结构：

![keras_fasttext](https://img-blog.csdnimg.cn/20200714204249463.jpg)

FastText在文本分类任务上，是优于TF-IDF的：

- FastText用单词的Embedding叠加获得的文档向量，将相似的句子分为一类
- FastText学习到的Embedding空间维度比较低，可以快速进行训练

如果想深度学习，可以参考论文：

Bag of Tricks for Efficient Text Classification, https://arxiv.org/abs/1607.01759

### 基于FastText的文本分类

FastText可以快速的在CPU上进行训练，最好的实践方法就是官方开源的版本：
https://github.com/facebookresearch/fastText/tree/master/python

- pip安装

```
pip install fasttext
```

- 源码安装

```
git clone https://github.com/facebookresearch/fastText.git
cd fastText
sudo pip install .
```

两种安装方法都可以安装，如果你是初学者可以优先考虑使用pip安装。

- 分类模型

```python
import pandas as pd
from sklearn.metrics import f1_score

# 转换为FastText需要的格式
train_df = pd.read_csv('../input/train_set.csv', sep='\t', nrows=15000)
train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
train_df[['text','label_ft']].iloc[:-5000].to_csv('train.csv', index=None, header=None, sep='\t')

import fasttext
model = fasttext.train_supervised('train.csv', lr=1.0, wordNgrams=2, 
                                  verbose=2, minCount=1, epoch=25, loss="hs")

val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-5000:]['text']]
print(f1_score(train_df['label'].values[-5000:].astype(str), val_pred, average='macro'))
# 0.82
```

此时数据量比较小得分为0.82，当不断增加训练集数量时，FastText的精度也会不断增加5w条训练样本时，验证集得分可以到0.89-0.90左右。

### 如何使用验证集调参

在使用TF-IDF和FastText中，有一些模型的参数需要选择，这些参数会在一定程度上影响模型的精度，那么如何选择这些参数呢？

- 通过阅读文档，要弄清楚这些参数的大致含义，那些参数会增加模型的复杂度
- 通过在验证集上进行验证模型精度，找到模型在是否过拟合还是欠拟合



![train_val](https://img-blog.csdnimg.cn/20200714204403844.png)

这里我们使用10折交叉验证，每折使用9/10的数据进行训练，剩余1/10作为验证集检验模型的效果。这里需要注意每折的划分必须保证标签的分布与整个数据集的分布一致。

```
label2id = {}
for i in range(total):
    label = str(all_labels[i])
    if label not in label2id:
        label2id[label] = [i]
    else:
        label2id[label].append(i)
```

通过10折划分，我们一共得到了10份分布一致的数据，索引分别为0到9，每次通过将一份数据作为验证集，剩余数据作为训练集，获得了所有数据的10种分割。不失一般性，我们选择最后一份完成剩余的实验，即索引为9的一份做为验证集，索引为1-8的作为训练集，然后基于验证集的结果调整超参数，使得模型性能更优。

### 本章小结

本章介绍了FastText的原理和基础使用，并进行相应的实践。然后介绍了通过10折交叉验证划分数据集。

### 本章作业

- 阅读FastText的文档，尝试修改参数，得到更好的分数
- 基于验证集的结果调整超参数，使得模型性能更优









