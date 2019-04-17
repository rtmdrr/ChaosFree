#!/usr/bin/env python3

import torch.nn.functional as F


def BNCFNCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):

  gi = F.linear(input, w_ih, b_ih)
  gh = F.linear(hidden, w_hh, b_hh)

  bn_gi = F.BatchNorm1d(gi, affine=False)
  bn_gh = F.BatchNorm1d(gh, affine=False)

  i_i, i_f, i_n = bn_gi.chunk(3, 1)
  h_i, h_f = bn_gh.chunk(2, 1)

  # f, i = sigmoid(Wx + Vh_tm1 + b)
  inputgate = F.sigmoid(i_i + h_i)
  forgetgate = F.sigmoid(i_f + h_f)
  newgate = i_n

  # h_t = f * tanh(h_tm1) + i * tanh(Wx)
  hy = inputgate * F.tanh(newgate) + forgetgate * F.tanh(hidden)

  return hy
