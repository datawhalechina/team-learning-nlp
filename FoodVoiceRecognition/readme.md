# 零基础入门语音识别之食物声音识别

## 赛题说明
本次新人赛是Datawhale与天池联合发起的0基础入门系列赛事 —— 零基础入门语音识别之食物声音识别。赛题以语音识别为背景，要求选手使用给定的音频数据集进行建模，并完成食物声音识别任务。为更好的引导大家入门，我们为本赛题定制了学习任务。通过对本方案的完整学习，可以帮助掌握数据分析基本技能。

## 数据集说明
数据集来自Kaggle的“[Eating Sound Collection](https://tianchi.aliyun.com/competition/entrance/531887/introduction)”（[可商用](https://opendatacommons.org/licenses/pddl/1-0/)），数据集中包含20种不同食物的咀嚼声音，赛题任务是给这些声音数据建模，准确分类。作为零基础入门语音识别的新人赛，本次任务不涉及复杂的声音模型、语言模型，希望大家通过两种baseline的学习能体验到语音识别的乐趣。

## 任务说明
- [Task1 食物声音识别之Baseline学习](https://github.com/datawhalechina/team-learning-nlp/tree/master/FoodVoiceRecognition/Task1%20%E9%A3%9F%E7%89%A9%E5%A3%B0%E9%9F%B3%E8%AF%86%E5%88%AB%E4%B9%8BBaseline%E5%AD%A6%E4%B9%A0)

  - 理解赛题、下载数据集以及两条Baseline（本次学习教程以基于CNN的Baseline为主）
  - 根据Baseline配置环境，也可以直接利用天池等环境运行
  - 跑通并学习Baseline

- [Task2 食物声音识别之赛题数据介绍与分析](https://github.com/datawhalechina/team-learning-nlp/blob/master/FoodVoiceRecognition/Task2%20%E9%A3%9F%E7%89%A9%E5%A3%B0%E9%9F%B3%E8%AF%86%E5%88%AB-%E8%B5%9B%E9%A2%98%E6%95%B0%E6%8D%AE%E4%BB%8B%E7%BB%8D%E4%B8%8E%E5%88%86%E6%9E%90.ipynb)

  - 赛题数据探索
  - 音频相关知识点学习

- [Task3 食物声音识别之音频数据特征提取](https://github.com/datawhalechina/team-learning-nlp/blob/master/FoodVoiceRecognition/Task3%20%E9%A3%9F%E7%89%A9%E5%A3%B0%E9%9F%B3%E8%AF%86%E5%88%AB-%E9%9F%B3%E9%A2%91%E6%95%B0%E6%8D%AE%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96.ipynb)

  - 学习多种音频数据特征
  - 了解MFCC特征提取步骤

- [Task4 食物声音识别之深度学习模型搭建、训练、验证](https://github.com/datawhalechina/team-learning-nlp/blob/master/FoodVoiceRecognition/Task4%20%E9%A3%9F%E7%89%A9%E5%A3%B0%E9%9F%B3%E8%AF%86%E5%88%AB-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A8%A1%E5%9E%8B%E6%90%AD%E5%BB%BA%E4%B8%8E%E8%AE%AD%E7%BB%83.ipynb)

  - 基于CNN的模型搭建、训练与验证
  - 了解CNN原理

- [Task5 食物声音识别之模型改进与优化](https://github.com/datawhalechina/team-learning-nlp/blob/master/FoodVoiceRecognition/Task5%20%E9%A3%9F%E7%89%A9%E5%A3%B0%E9%9F%B3%E8%AF%86%E5%88%AB-%E6%A8%A1%E5%9E%8B%E6%94%B9%E8%BF%9B%E4%B8%8E%E4%BC%98%E5%8C%96.ipynb)

  - 学习模型优化相关知识
  - 自己尝试基于Baseline的模型进行优化或尝试其他模型以提升结果准确率


- [Task6 拓展阅读：语音识别基础知识介绍](https://github.com/datawhalechina/team-learning-nlp/blob/master/FoodVoiceRecognition/Task6%20%E6%8B%93%E5%B1%95%E9%98%85%E8%AF%BB%EF%BC%9A%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86%E4%BB%8B%E7%BB%8D.ipynb)

  - 了解语音识别的基础背景知识

## 贡献者信息
| 姓名                                                         | 介绍                                                         | 个人主页                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 阿水 |Datawhale成员 | 公众号：Coggle数据科学    |
| 黎佳佳 |Datawhale成员 |  公众号：ICE的小窝   |
| 但扬杰 | 江西师范大学软件工程硕士，Datawhale成员 |  [github账号](https://github.com/jianghusanren007)    |
| 陈安东 | 中央民族大学，Datawhale成员 |   [知乎主页](https://www.zhihu.com/people/wang-ya-fei-48)    |
| 付文豪 |Datawhale优秀学习者   |         |
| 马琦钧 |Datawhale成员 |       |

## 项目贡献情况

项目构建与整合：阿水、黎佳佳

task1：陈安东、但扬杰

task2：黎佳佳

task3：陈安东、黎佳佳

task4：陈安东

task5：付文豪

task6：马琦钧


## 致谢
- [语音识别基本方法](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzI2MzU4NDI4NA==&action=getalbum&album_id=1472128841614753794&scene=173&from_msgid=2247484000&from_itemidx=1&count=3#wechat_redirect)
- [微软Edx语音识别课程](http://fancyerii.github.io/2019/05/25/dev287x/)
- [使用带有Keras的卷积神经网络的城市声音分类](https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4)

## 关注我们
<div align=center><img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "250" height = "270" alt="Datawhale是一个专注AI领域的开源组织，以“for the learner，和学习者一起成长”为愿景，构建对学习者最有价值的开源学习社区。关注我们，一起学习成长。"></div>
