"""VCOPN"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple


class VCOPN(nn.Module):
    """Video clip order prediction with PFE (Pairwire Feature Extraction), the same as OPN."""
    def __init__(self, base_network, feature_size, tuple_len):
        """
        Args:
            feature_size (int): 512
        """
        super(VCOPN, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)

        self.fc7 = nn.Linear(self.feature_size*2, 512)
        pair_num = int(tuple_len*(tuple_len-1)/2)
        self.fc8 = nn.Linear(512*pair_num, self.class_num)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, tuple):
        f = []  # clip features
        for i in range(self.tuple_len):
            clip = tuple[:, i, :, :, :, :]
            f.append(self.base_network(clip))

        pf = []  # pairwise concat
        for i in range(self.tuple_len):
            for j in range(i+1, self.tuple_len):
                pf.append(torch.cat([f[i], f[j]], dim=1))

        pf = [self.fc7(i) for i in pf]
        pf = [self.relu(i) for i in pf]
        h = torch.cat(pf, dim=1)
        h = self.dropout(h)
        h = self.fc8(h)  # logits

        return h


class VCOPN_RNN(nn.Module):
    """Video clip order prediction with RNN."""
    def __init__(self, base_network, feature_size, tuple_len, hidden_size, rnn_type='LSTM'):
        """
        Args:
            feature_size (int): 1024
        """
        super(VCOPN_RNN, self).__init__()

        self.base_network = base_network
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.tuple_len = tuple_len
        self.class_num = math.factorial(tuple_len)
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.lstm = nn.LSTM(self.feature_size, self.hidden_size)
        elif self.rnn_type == 'GRU':
            self.gru = nn.GRU(self.feature_size, self.hidden_size)
        
        self.fc = nn.Linear(self.hidden_size, self.class_num)

    def forward(self, tuple):
        f = []  # clip features
        for i in range(self.tuple_len):
            clip = tuple[:, i, :, :, :, :]
            f.append(self.base_network(clip))

        inputs = torch.stack(f)
        if self.rnn_type == 'LSTM':
            outputs, (hn, cn) = self.lstm(inputs)
        elif self.rnn_type == 'GRU':
            outputs, hn = self.gru(inputs)

        h = self.fc(hn.squeeze(dim=0))  # logits

        return h
