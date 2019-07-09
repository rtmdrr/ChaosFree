import torch
from torch import nn
import torch.nn.functional as F
import math

class BNCFNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
            super(BNCFNCell, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.w_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
            self.w_hh = nn.Parameter(torch.Tensor(2 * hidden_size, hidden_size))
            self.b_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.b_hh = nn.Parameter(torch.Tensor(2 * hidden_size))

            self.bn_ih = nn.BatchNorm1d(3 * self.hidden_size, affine=False)
            self.bn_hh = nn.BatchNorm1d(2 * self.hidden_size, affine=False)

            self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        

    def forward(self, input, h):
        gi = F.linear(input, self.w_ih, self.b_ih)
        gh = F.linear(h, self.w_hh, self.b_hh)
        bn_gi = self.bn_ih(gi)
        bn_gh = self.bn_hh(gh)
        i_i, i_f, i_n = bn_gi.chunk(3, 1)
        h_i, h_f = bn_gh.chunk(2, 1)

        # f, i = sigmoid(Wx + Vh_tm1 + b)
        inputgate = torch.sigmoid(i_i + h_i)
        forgetgate = torch.sigmoid(i_f + h_f)
        newgate = i_n

        # h_t = f * tanh(h_tm1) + i * tanh(Wx)
        hy = inputgate * torch.tanh(newgate) + forgetgate * torch.tanh(h)

        return hy


class BNCFN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False, bidirectional=False, dropout=False):
        super(BNCFN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.cfn_f = BNCFNCell(input_size, hidden_size)
        if bidirectional:
            self.cfn_b = BNCFNCell(input_size, hidden_size)
        self.h0 = nn.Parameter(torch.Tensor(2 if self.bidirectional else 1, 1, self.hidden_size))
        nn.init.normal_(self.h0, mean=0, std=0.1)

    def forward(self, input, hx=None):
        if not self.batch_first:
            input = input.transpose(0, 1)
        batch_size, seq_len, dim = input.size()
        if hx is not None:
            init_state = hx
        else:
            init_state = self.h0.repeat(1, batch_size, 1)
        
        hiddens_f = []
        final_hx_f = None
        hx = init_state[0]
        for i in range(seq_len):
            hx = self.cfn_f(input[:, i, :], hx)
            hiddens_f.append(hx)
            final_hx_f = hx
        hiddens_f = torch.stack(hiddens_f, 1)
        
        if self.bidirectional:
            hiddens_b = []
            final_hx_b = None
            hx = init_state[1]
            for i in range(seq_len-1, -1, -1):
                hx = self.cfn_b(input[:, i, :], hx)
                hiddens_b.append(hx)
                final_hx_b = hx
            hiddens_b.reverse()
            hiddens_b = torch.stack(hiddens_b, 1)
        
        if self.bidirectional:
            hiddens = torch.cat([hiddens_f, hiddens_b], -1)
            hx = torch.stack([final_hx_f, final_hx_b], 0)
        else:
            hiddens = hiddens_f
            hx = hx.unsqueeze(0)
        if not self.batch_first:
            hiddens = hiddens.transpose(0, 1)
        return hiddens, hx