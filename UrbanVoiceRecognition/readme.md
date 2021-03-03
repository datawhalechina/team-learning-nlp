# 零基础入门语音识别之食物声音识别

## 语音识别基础知识
语音识别全称为“自动语音识别”，Automatic Speech Recognition (ASR), 一般是指将语音序列转换成文本序列。语音识别最终是统计优化问题，给定输入序列O={O1,...,On}，寻找最可能的词序列W={W1,...,Wm}，即寻找使得概率P(W|O)最大的词序列。用贝叶斯公式表示为：
![](https://latex.codecogs.com/gif.latex?\dpi{400}\alpha&space;+&space;\frac{2\beta}{\gamma})
