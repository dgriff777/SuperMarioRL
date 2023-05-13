from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import norm_col_init, weights_init
import torch.autograd.profiler as profiler
from torch.nn.functional import normalize
import math


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)


    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 3, 2, 1)
        res_input = x
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x += res_input
        res_input = x
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x += res_input
        return x


class MarioNET(nn.Module):
    def __init__(self, num_inputs, action_space, args):
        super(MarioNET, self).__init__()
        self.hidden_size = args.hidden_size
        self.num_actions = action_space.n
        input_channels = num_inputs
        self.resnet_blocks = []
        for num_ch in [16, 32, 32]:
            self.resnet_blocks.append(ResBlock(input_channels, num_ch))
            input_channels = num_ch
        self.resnet_blocks = nn.ModuleList(self.resnet_blocks)
        self.fc = nn.Linear(3200, self.hidden_size)
        self.lstm = nn.LSTMCell(self.hidden_size, self.hidden_size)
        core_output_size = self.hidden_size
        self.actor_linear = nn.Linear(core_output_size, self.num_actions)
        self.critic_linear = nn.Linear(core_output_size, 1)
        self.apply(weights_init)
        self.actor_linear.weight.data = norm_col_init(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = norm_col_init(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)
        relu = nn.init.calculate_gain("relu")
        self.fc.weight.data.mul_(relu)
        self.fc.bias.data.fill_(0)
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, p in self.named_parameters():
            if "lstm" in name:
                if "weight_ih" in name:
                    nn.init.uniform_(p.data, -stdv, stdv)
                elif "weight_hh" in name:
                    nn.init.uniform_(p.data, -stdv, stdv)
                elif "bias_ih" in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif "bias_hh" in name:
                    p.data.fill_(0)
        self.train()


    def forward(self, inputs, hx, cx):
        x = inputs
        for resnet_block in self.resnet_blocks:
            x = resnet_block(x)
        x = F.relu(x)
        x = x.view(1, 3200)
        x = F.relu(self.fc(x))
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.critic_linear(x), self.actor_linear(x), hx, cx
