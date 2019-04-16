#!/usr/bin/env python3

import torch.nn.functional as F


def CFNCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):

  gi = F.linear(input,w_ih, b_ih)
  gh = F.linear(hidden, w_hh, b_hh)

  i_i, i_f, i_n = gi.chunk(3, 1)
  h_i, h_f = gh.chunk(2, 1)

  # f, i = sigmoid(Wx + Vh_tm1 + b)
  inputgate = F.sigmoid(i_i + h_i)
  forgetgate = F.sigmoid(i_f + h_f)
  newgate = i_n

  # h_t = f * tanh(h_tm1) + i * tanh(Wx)
  hy = inputgate * F.tanh(newgate) + forgetgate * F.tanh(hidden)

  return hy
