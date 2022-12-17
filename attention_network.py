import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class AttNet(nn.Module):
    def __init__(self, var_size, attention_size, k, hidden_size):
        super(AttNet, self).__init__()
        self.var_size = var_size
        self.attentio_size = attention_size
        self.k = k
        self.hidden_size = hidden_size
        self.constraint_lstm_embedding = nn.LSTM(var_size, attention_size, batch_first=True)
        self.candidate_lstm_embedding = nn.LSTM(var_size, attention_size, batch_first=True)

        
        self.F_constraint = torch.nn.Sequential(
                    torch.nn.Linear(self.attentio_size, self.hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(self.hidden_size, self.hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(self.hidden_size, self.k)
        )
                
        self.F_candidate = torch.nn.Sequential(
                    torch.nn.Linear(self.attentio_size, self.hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(self.hidden_size, self.hidden_size),
                    torch.nn.Tanh(),
                    torch.nn.Linear(self.hidden_size, self.k)
        )

    def forward(self, constraint, candidate):
        constraint = torch.Tensor(np.expand_dims(constraint,1))
        candidate = torch.Tensor(np.expand_dims(candidate,1))
        out_a_b,(hid_a_b, _) = self.constraint_lstm_embedding(constraint)
        our_e_d,(hid_e_d, _) = self.candidate_lstm_embedding(candidate)
        a_b_embedding = torch.squeeze(hid_a_b)
        e_d_embedding = torch.squeeze(hid_e_d)
        H = self.F_constraint(a_b_embedding)
        G = self.F_candidate(e_d_embedding)
        S = G @ torch.transpose(H, 0, 1)
        Attention_score = torch.mean(S, dim = 1)
        return Attention_score
