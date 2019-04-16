#!/usr/bin/env python3

import torch.nn as nn
import torch as T
from torch.nn.modules.rnn import RNNCellBase
from CFN.functional import CFNCell as CFNCellF

import math


class CFNCell(RNNCellBase):
  r"""
  """

  def __init__(self, input_size, hidden_size, bias=True):
    super(CFNCell, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bias = bias

    self.weight_ih = nn.Parameter(T.Tensor(3 * hidden_size, input_size))
    self.weight_hh = nn.Parameter(T.Tensor(2 * hidden_size, hidden_size))
    if self.bias:
      self.bias_ih = nn.Parameter(T.Tensor(3 * hidden_size))
      self.bias_hh = nn.Parameter(T.Tensor(2 * hidden_size))
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
      weight.data.uniform_(-stdv, stdv)

  def forward(self, input, hx):
    return CFNCellF(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
