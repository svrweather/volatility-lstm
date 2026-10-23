import math
import torch
import torch.nn as nn

class MyLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.empty(4 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.empty(4 * hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        k = 1 / math.sqrt(self.hidden_dim)
        for p in self.parameters():
            nn.init.uniform_(p, -k, k)

    def forward(self, x, h, c):
        gates = (x @ self.weight_ih.T + self.bias_ih + 
                 h @ self.weight_hh.T + self.bias_hh)

        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)    # что записать
        f = torch.sigmoid(f)    # что забыть
        g = torch.tanh(g)       # кандидат на запись
        o = torch.sigmoid(o)    # что выдать

        c = c * f + g * i
        h = torch.tanh(c) * o
        return h, c

class MyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            MyLSTMCell(input_dim if l == 0 else hidden_dim, hidden_dim)
            for l in range(num_layers)
        ])

    def forward(self, x, state=None):
        # x (batch, seq_len, inp_dim)
        batch, seq_len, _ = x.shape
        if state is None:
            h0 = x.new_zeros(self.num_layers, batch, self.hidden_dim)
            c0 = x.new_zeros(self.num_layers, batch, self.hidden_dim)
        else:
            h0, c0 = state
        layer_input = x
        h_n, c_n = [], []
        for l, cell in enumerate(self.cells):
            h, c = h0[l], c0[l]
            outputs = []
            for t in range(seq_len):
                h, c = cell(layer_input[:, t, :], h, c)
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=1) # (batch, seq_len, hid_dim)
            h_n.append(h)
            c_n.append(c)
        out = layer_input
        return out, (torch.stack(h_n), torch.stack(c_n))