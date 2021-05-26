from collections import defaultdict
import codecs
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os

class CharLSTM(nn.Module):
    def __init__(self, latent_size, alph_size=256, n_layers=2):
        super(CharLSTM, self).__init__()
        self.latent_size = latent_size
        self.alph_size = alph_size
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size = alph_size,
                          hidden_size = latent_size,
                          num_layers = 2,
                          batch_first = True)
        self.lin = nn.Linear(latent_size, alph_size,bias = False)

    def forward(self, x, state):
        x = x.to(torch.float32)
        for s in state:
            s = s.to(torch.float32)
        out, state = self.rnn(x, state)
        out = self.lin(out.flatten(end_dim =1))
        return out, state

    def predict(self, x, state):
        x, state = self.forward(x, state)
        return F.softmax(x, dim=0), state
    
    def init_state(self):
        return (torch.zeros(self.n_layers,1,net.latent_size),
               torch.zeros(self.n_layers,1,net.latent_size))

def tokenize(x, length):
    return F.one_hot(x, length)

class ExperienceReplay(Dataset):
    def __init__(self, context_len=8, alph_len=256, max_size = 5000):
        super(ExperienceReplay, self).__init__()
        self.data = list()
        self.alph_len = alph_len
        self.ctx_len = context_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.tokenize(self.data[idx], self.alph_len)
    
    def get_batch(self):
        idxs = torch.randperm(self.__len__())
        return torch.stack(
            [self.tokenize(self.data[idx], self.alph_len) for idx in idxs],
            dim = 0)
    
    def insert(self, context):
        self.data.append(torch.LongTensor([ord(c) for c in context]))
    
    def forget(self, idx):
        self.data.pop(idx)
        
