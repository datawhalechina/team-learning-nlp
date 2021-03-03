# 零基础入门语音识别之食物声音识别

## 语音识别基础知识

语音识别全称为“自动语音识别”，Automatic Speech Recognition (ASR), 一般是指将语音序列转换成文本序列。语音识别最终是统计优化问题，给定输入序列O={O1,...,On}，寻找最可能的词序列W={W1,...,Wm}，即寻找使得概率P(W|O)最大的词序列。用贝叶斯公式表示为：![](https://latex.codecogs.com/gif.latex?\\P(W|O)=\frac{P(O|W)P(W)}{P(O)})

其中P(O|W) 叫做声学模型，描述的是给定词W时声学观察为O的概率；P(W)叫做语言模型，负责计算某个词序列的概率；P(O)是观察序列的概率，是固定的，是固定的，所以只看分母部分即可。

语音选择的基本单位是帧（Frame），一帧数据是由一小段语音经过ASR前端的声学特征提取模块产生的，整段语音就可以整理为以帧为单位的向量组。每帧的维度固定不变，但跨度可调，以适应不同的文本单位，比如音素、字、词、句子。

大多数语音识别的研究都是分别求取声学和语言模型，并把很多精力放在声学模型的改进上。但后来，基于深度学习和大数据的端到端（End-to-End）方法发展起来，能将声学和语言模型融为一体，直接计算P(W|O)。

## 赛题说明
本次新人赛是Datawhale与天池联合发起的0基础入门系列赛事 —— 零基础入门语音识别之食物声音识别。赛题以语音识别为背景，要求选手使用给定的音频数据集进行建模，并完成食物声音识别任务。为更好的引导大家入门，我们为本赛题定制了学习任务。通过对本方案的完整学习，可以帮助掌握数据分析基本技能。

## 数据集说明
数据集来自Kaggle的“[Eating Sound Collection](https://www.kaggle.com/mashijie/eating-sound-collection)”（[可商用](https://opendatacommons.org/licenses/pddl/1-0/)），数据集中包含20种不同食物的咀嚼声音，赛题任务是给这些声音数据建模，准确分类。

作为零基础入门语音识别的新人赛，本次任务不涉及复杂的声音模型、语言模型，希望大家通过两种baseline的学习能体验到语音识别的乐趣。

## 任务说明
我们提供了两种Baseline供大家学习。
- Task1 基于LSTM的分类模型：通过数据预处理、特征提取、划分数据集以及训练模型等步骤给声音数据做分类。
- Task2 基于CNN的分类模型：参考图片分类的形式，将不同声音的频谱做分类。这种形式虽然不能准确识别人说话的文本，但对于本次任务中区分不同类别的声音任务是足够的。

## 贡献者信息
| 姓名                                                         | 介绍                                                         | 个人主页                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 阿水 |Datawhale成员 |     |
| 黎佳佳 |Datawhale成员 |  公众号：ICE的小窝   |
| 旦扬杰 | Datawhale成员 |       |
| 陈安东 | Datawhale成员 |       |
| 付文豪 | Datawhale成员  |         |
| 马琦钧 |Datawhale成员 |       |

## 致谢
- [语音识别基本方法](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI2MzU4NDI4NA==&action=getalbum&album_id=1472128841614753794&scene=173&from_msgid=2247484000&from_itemidx=1&count=3#wechat_redirect)
- [微软Edx语音识别课程](http://fancyerii.github.io/2019/05/25/dev287x/)
- [使用带有Keras的卷积神经网络的城市声音分类](https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4)
