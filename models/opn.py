"""OPN"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class OPN(nn.Module):
    """Frame Order Prediction Network"""
    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 256
        """
        super(OPN, self).__init__()

        self.base_network = base_network
        self.tuple_len = tuple_len
        self.feature_size = feature_size
        self.class_num = math.factorial(tuple_len)

        self.fc7 = nn.Linear(self.feature_size*2, 512)
        pair_num = int(tuple_len*(tuple_len-1)/2)
        self.fc8 = nn.Linear(512*pair_num, self.class_num)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, tuple):
        f = []  # frame features
        for i in range(self.tuple_len):
            frame = tuple[:, i, :, :, :]
            f.append(self.base_network(frame))

        pf = []  # pairwise features
        for i in range(self.tuple_len):
            for j in range(i+1, self.tuple_len):
                pf.append(torch.cat([f[i], f[j]], dim=1))

        pf = [self.fc7(i) for i in pf]
        pf = [self.relu(i) for i in pf]
        h = torch.cat(pf, dim=1)
        h = self.dropout(h)
        h = self.fc8(h) # logits

        return h


class OPN_RNN(nn.Module):
    """Frame Order Prediction Network with RNN"""
    def __init__(self, base_network, feature_size, tuple_len, hidden_size, rnn_type='LSTM'):
        """
        Args:
            feature_size (int): 256
        """
        super(OPN_RNN, self).__init__()

        self.base_network = base_network
        self.tuple_len = tuple_len
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.class_num = math.factorial(tuple_len)

        if self.rnn_type == 'LSTM':
            self.lstm = nn.LSTM(self.feature_size, self.hidden_size, num_layers=2, bidirectional=True)
        elif self.rnn_type == 'GRU':
            self.gru = nn.GRU(self.feature_size, self.hidden_size, num_layers=2, bidirectional=True)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(self.feature_size, self.hidden_size, num_layers=2, bidirectional=True, nonlinearity = 'relu')

        self.fc = nn.Linear(self.hidden_size*2, self.class_num)

    def forward(self, tuple):
        f = []  # frame features
        for i in range(self.tuple_len):
            frame = tuple[:, i, :, :, :]
            f.append(self.base_network(frame))

        inputs = torch.stack(f)
        if self.rnn_type == 'LSTM':
            outputs, (hn, cn) = self.lstm(inputs)
        elif self.rnn_type == 'GRU':
            outputs, hn = self.gru(inputs)
        elif self.rnn_type == 'RNN':
            outputs, hn = self.rnn(inputs)

        h = self.fc(outputs[-1])  # logits

        return h
