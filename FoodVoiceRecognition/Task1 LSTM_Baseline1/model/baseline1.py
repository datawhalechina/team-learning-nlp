import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline1(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, args):
        super(Baseline1, self).__init__()
        self.args = args
        self.d_input = self.args.d_input
        self.class_num = self.args.class_num

        self.hidden_size = 128
        self.bn_dim = 30
        self.num_layers = 1

        self.lstm = nn.LSTM(input_size=self.d_input, hidden_size=self.hidden_size, num_layers=self.num_layers,
                            bidirectional=False, bias=True, batch_first=True)

        self.linear_layer1 = nn.Linear(self.hidden_size, self.bn_dim)
        self.linear_layer2 = nn.Linear(self.bn_dim, self.class_num)

    def forward(self, encoder_out, input_lengths):
        # 语谱图语谱图
        #x = encoder_out.reshape(encoder_out.shape[0],encoder_out.shape[1]*encoder_out.shape[2],encoder_out.shape[3])
        x = self.lstm(encoder_out)[0]
        x = self.linear_layer1(x)
        x = self.linear_layer2(x)
        x = torch.mean(x, dim=1)
        return x,






