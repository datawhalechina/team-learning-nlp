import sys
sys.path.append('../')

import argparse
import torch
import os
import numpy as np
from data.data import AudioDataLoader, AudioDataset
from solver.solver import Solver
from model.baseline1 import Baseline1

parser = argparse.ArgumentParser("Datawhale NLP")

# General config
# Task related

parser.add_argument('--train-mfccjson', type=str, default='../dump/data_train.json',
                    help='Filename of train label data (json)')
parser.add_argument('--valid-mfccjson', type=str, default='../dump/data_dev.json',
                    help='Filename of validation label data (json)')


# Low Frame Rate (stacking and skipping frames)
parser.add_argument('--LFR_m', default=1, type=int,
                    help='Low Frame Rate: number of frames to stack') #4
parser.add_argument('--LFR_n', default=1, type=int,
                    help='Low Frame Rate: number of frames to skip') #3

#model

parser.add_argument('--class_num', default=20, type=int,    # 这里这里这里#
                    help='num of dialects class')
parser.add_argument('--d_input', default=60, type=int,      # 这里这里这里#
                    help='Dim of encoder input (before LFR)')
parser.add_argument('--n_layers', default=1, type=int,    # 这里这里这里#
                    help='num of dialects class')

# save and load model
parser.add_argument('--save-folder', default='exp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue-from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model-path', default='mfcc_datawhale.pth.tar',   # 这里这里这里#
                    help='Location to save best validation model')

# logging
parser.add_argument('--print-freq', default=100, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom_lr', dest='visdom_lr', type=int, default=0,
                    help='Turn on visdom graphing learning rate')
parser.add_argument('--visdom_epoch', dest='visdom_epoch', type=int, default=0,
                    help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom-id', default='Transformer training',
                    help='Identifier for visdom run')

# Training config
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of maximum epochs')

parser.add_argument('--lr', default=0.0005, type=float,
                    help='learn rate')
# minibatch
parser.add_argument('--shuffle', default=1, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch-size', default=16, type=int,
                    help='Batch size')
parser.add_argument('--batch_frames', default=10000, type=int,  #20000
                    help='Batch frames. If this is not 0, batch size will make no sense') #15000 10000
parser.add_argument('--maxlen-in', default=800, type=int, metavar='ML',
                    help='Batch size is reduced if the input sequence length > ML')
parser.add_argument('--maxlen-out', default=150, type=int, metavar='ML',
                    help='Batch size is reduced if the output sequence length > ML')
parser.add_argument('--num-workers', default=0, type=int,
                    help='Number of workers to generate minibatch')



parser.add_argument('--model_choose', default='baseline', type=str,  # 这里这里这里#
                    help='choose model type')
#baseline4
parser.add_argument('--layer_choose', default='layer1', type=str,
                    help='choose model type')  #layer1, layer2, layer3, layer4, all

def main(args):
    # Construct Solver
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    # data
    tr_dataset = AudioDataset(args.train_mfccjson, args.batch_size,
                              args.maxlen_in, args.maxlen_out,
                              batch_frames=args.batch_frames)
    cv_dataset = AudioDataset(args.valid_mfccjson,  args.batch_size,
                              args.maxlen_in, args.maxlen_out,
                              batch_frames=args.batch_frames)
    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                num_workers=args.num_workers,
                                shuffle=args.shuffle,
                                LFR_m=args.LFR_m, LFR_n=args.LFR_n, model_choose=args.model_choose)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                num_workers=args.num_workers,
                                LFR_m=args.LFR_m, LFR_n=args.LFR_n, model_choose=args.model_choose)


    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    model = Baseline1(args)

    print(model)
    print('model parameters:', sum(param.numel() for param in model.parameters()))
    model.cuda()

    optimizier = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), betas=(0.9, 0.98), eps=1e-09,
                                  lr=args.lr)

    # solver
    solver = Solver(data, model, optimizier, args)
    solver.train()




if __name__ == '__main__':


    args = parser.parse_args()
    print(args)
    main(args)
