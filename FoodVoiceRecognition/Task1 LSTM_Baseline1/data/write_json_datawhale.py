import json
from collections import OrderedDict
import collections
import os
import numpy as np

def write_mydata_json():
    """
    1 函数功能： 编写最终的data_train.json, data_dev.json,适用于非kaldi(自己编写的特征提取脚本)
    :return:
    """
    wav_path = '../clips_rd_sox'
    mfcc_path = '../clips_rd_mfcc'
    train_json_dir = '../dump/data_train.json'
    dev_json_dir = '../dump/dev_train.json'

    train_dict = collections.OrderedDict()
    dev_dict = collections.OrderedDict()

    for root, dirs, files in os.walk(wav_path):
        if files:
            train_num = int(len(files) * 0.8)
            for index, file in enumerate(files):
                featrue_path = mfcc_path + file.replace('wav', 'npy')
                shape_num = list(np.load(featrue_path).shape)
                input_dict = {'feat': '../clips_rd_mfcc/{}'.format(file.replace('wav', 'npy')),
                              'shape': shape_num,
                              }
                if index <= train_num:
                    train_dict.update({file:input_dict})
                else:
                    dev_dict.update({file:input_dict})

    with open(train_json_dir, 'w') as f:
        json.dump(train_dict, f, indent=4)
        print("加载入文件完成...")

    with open(dev_json_dir, 'w') as f:
        json.dump(dev_dict, f, indent=4)
        print("加载入文件完成...")


