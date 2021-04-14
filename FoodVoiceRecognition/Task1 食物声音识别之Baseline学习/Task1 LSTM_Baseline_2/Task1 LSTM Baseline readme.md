# Baseline运行细节说明

## 任务说明

本次学习任务属于分类任务，数据集包含了咀嚼20种不同食物的声音音频文件，任务目的是给定一段音频文件，能够正确识别出所属类别。

## 数据集

本次学习任务数据集为咀嚼20种不同食物的声音音频(wav文件)，每个类别大概有500条wav文件，总计11140条wav文件。

## 基线模型

Baseline我们选择的是LSTM，评价指标是ACC(准确率)，损失函数是交叉熵。

## 文件夹介绍

总文件夹名称为src_datawhale，里面有7个文件夹，分别是：

- bin：里面的train.py是总运行脚本，当一切准备就绪后运行此脚本可以得到结果；exp文件夹是用来存放训练好的模型和结果(结果需要自己手动复制到创建的文件中)。
- clips_rd_mfcc：里面是11140个npy文件，对应着11140条wav文件的mfcc特征数据。
- data：里面有3个py文件和一个sh文件，其中extract_mfcc_datawhale.py，write_json_datawhale.py和sox_audio.sh是单独运行的(即分别运行这3个文件会得到我们需要的准备数据)，data.py是本系统的一个配套文件(得到前面提到的所需要的准备数据后，再运行train.py)。
- dump：里面是两个json文件，是用来保存训练集和测试集的相关信息的(包括wav数据的名字，mfcc特征形状，wav文件对应的mfcc特征文件路径)。
- model：里面存放的是我们的模型py文件，这里我们选择的是LSTM。
- solver：此文件是运行的细节，比如epoch大小，保存训练得到的最佳性能下的模型，打印训练结果等。
- utils：里面是一个utils.py，是自己定义的一些功能函数集合。

## 运行环境

这里列出几个重要的环境(其他一些依赖库安装非常简单，这里就没有提及)，大家需要安装：

python35 + torch1.1 + torchvision0.3 + cuda9.0，运行系统是ubuntu16

笔者用的是上述的环境，也许有的同学版本高于笔者版本，具体其他版本大家可以自己尝试一下

提供一个安装命令(如果不可用试试其他方法)：

conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch

## 运行流程

这里我提供一个运行流程，希望大家在不研究代码的情况下能快速运行一下，运行所占显存760M左右，时间在5-10分钟左右。大家可以按照下面的顺序依次执行相关程序：

1. 运行sox_audio.sh：

   此文件功能是将原来wav数据集进行规整成16bit，16000的采样率的wav文件，之所以有这么一步操作是因为笔者在用librosa进行mfcc提取的时候发现有近一半的wav文化无法提取。经过此操作后都能正常提取，此sh文件里面涉及到的工具sox是在ubuntu系统下运行的。不想进行此步骤的同学可以用我们处理好的wav文件代替原始数据集即可。

   ```
   bash sox_audio.sh
   ```

   运行的时候请将此sh文件和原始数据集放在同一个目录下，运行结束后将会生成一个名字为“a”的文件夹，再把a文件夹里面的数据剪切出来代替原来的数据即可，笔者是将处理完的数据集放在了文件夹clips_rd_sox下。
   Windows系统下，如果无法运行该文件，可以直接在云盘下载已经处理好的数据集并放到与bin文件夹的同级目录下即可，clips_rd_sox：https://pan.baidu.com/s/1K2vgn_Wf8Foy3oYHPH22LA 提取码：4vdz 


2. 运行extract_mfcc_datawhale.py：

   此文件功能是提取mfcc特征，运行结束后会将提取的.npy文件存储到clips_rd_mfcc文件夹下。

   ```python
   python extract_mfcc_datawhale.py 或者打开pycharm点击运行
   ```

   

3. 运行write_json_datawhale.py：

   此文件功能是划分数据集，并且将训练集和测试集的相关信息写入json文件中。（后续data.py读取需要用到这两个json文件）。

   ```
   python write_json_datawhale.py 或者打开pycharm点击运行
   ```
   写入json中的信息种可以看到，从中我们可以得到名字、mfcc特征向量的形状以及mfcc特征对应npy文件的路径。这里需要说明的是shape的作用：在图片领域，我们选取一个batch往往都可以固定图片的数量，比如100张图片作为一个batch去训练。但是在语音领域不同，不能拿100条wav数据作为一个batch，因为语音有长有短，对应的帧数有大有小，例如上图中第一个为367帧，第二个为173帧。所以笔者选择batch的标准是累计达到10000帧(小于或者等于10000)所对应的wav文件作为一个batch。

4. 运行train.py：

   此文件是总运行程序，在前面的准备工作完成后，运行即可得到打印的结果。

   ```
   python train.py 或者打开pycharm点击运行
   ```


## 特别说明

因为本系统考虑的东西较为完善，所以写的比较复杂，有些代码对整体没有很大作用但也写上去了，例如solver.py中的def _reset(self)函数功能是加载上次训练的模型继续训练，以免从头开始训练浪费时间。但是本baseline训练速度很短，因此没用到此功能；再例如train.py中定义参数parser.add_argument，有一些参数后面没有用到；再例如data.py中的def build_LFR_features(inputs, m, n)功能是堆叠m帧以及跳n帧，因为语音是连续的，堆叠m帧的mfcc特征说不定效果更好。当然，此处我们将m，n都取为1，所以也没有用到此功能，有兴趣的同学可以进行不同的尝试；再例如extract_mfcc_datawhale.py中提供了提取两种特征(mfcc，log-mel)函数，大家可以选择其一或者笔者没有提及的特征，例如fbank。

本项目参考项目《Speech Transformer: End-to-End ASR with Transformer》进行修改而来，感谢原项目作者大大开源的优秀项目。 Reference https://github.com/zhaoyi2/Speech-Transformer
