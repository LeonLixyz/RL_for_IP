import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import attention_network


class Policy_Network(object):
    
    def __init__(self, var_size, attention_size, k, hidden_size, lr):
        
        self.ATT_net = attention_network.AttNet(var_size = var_size, attention_size = attention_size, k = k, hidden_size = hidden_size)
        
        # DEFINE THE OPTIMIZER
        self.optimizer = torch.optim.Adam(self.ATT_net.parameters(), lr=lr)
        
        # RECORD HYPER-PARAMS
        self.attention_size = attention_size
        self.k = k
        self.hidden_size = hidden_size
            
    def compute_attention(self, constraint,candidate):
        return self.ATT_net(constraint, candidate)
        
    def compute_prob(self, attention):
        prob = torch.nn.functional.softmax(attention, dim=0)
        return prob.cpu().data.numpy()
    
    def _to_one_hot(self, y, num_classes):
        """
        convert an integer vector y into one-hot representation
        """
        scatter_dim = len(y.size())
        y_tensor = y.view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)
        return zeros.scatter(scatter_dim, y_tensor, 1)
    
    def train(self, constraint, candidate, act, q_s):

        
        Qs = torch.FloatTensor(q_s)
        act = torch.LongTensor(act)
        prob = torch.nn.functional.softmax(self.ATT_net(constraint, candidate), dim=-1)
        prob_selected = torch.sum(self._to_one_hot(act, len(prob)) * prob, -1)
        
        # FOR ROBUSTNESS
        prob_selected += 1e-8

        # TODO define loss function as described in the text above
        loss = -torch.mean(torch.log(prob_selected) * Qs)

        # BACKWARD PASS
        self.optimizer.zero_grad()
        loss.backward()

        # UPDATE
        self.optimizer.step()
            
        return loss.detach().cpu().data.numpy()
