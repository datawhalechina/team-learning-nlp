#!/bin/sh

python main.py  --task_name GINGraphPooling\    # 为当前试验取名
                --device 0\                     
                --num_layers 5\                 # 使用GINConv层数
                --graph_pooling sum\            # 图读出方法
                --emb_dim 256\                  # 节点嵌入维度
                --drop_ratio 0.\
                --save_test\                    # 是否对测试集做预测并保留预测结果
                --batch_size 512\
                --epochs 100\
                --weight_decay 0.00001\
                --early_stop 10\                # 当有`early_stop`个epoches验证集结果没有提升，则停止训练
                --num_workers 4\
                --dataset_root dataset          # 存放数据集的根目录

