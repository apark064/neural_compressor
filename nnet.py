from collections import defaultdict
import codecs
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import os

class CharLSTM(nn.Module):
    def __init__(self, latent_size, alph_len=256, **kwargs):
        super(CharLSTM, self).__init__()
        self.latent_size = latent_size
        self.alph_len = alph_len
        self.num_layers = kwargs.get("num_layers",2)
        self.rnn = nn.LSTM(input_size = alph_len,
                          hidden_size = latent_size,
                          num_layers = self.num_layers,
                          batch_first = True)
        self.lin = nn.Linear(latent_size, alph_len, bias = True)

    def forward(self, x, state):
        x = x.to(torch.float32)
        for s in state:
            s = s.to(torch.float32)
        out, state = self.rnn(x, state)
        out = self.lin(out.flatten(end_dim =1))
        return out, state

    def predict(self, context, state = None):
        if state is None:
            state = self.init_state()
        out, state = self.forward(context.unsqueeze(0), state)
        return F.softmax(out[-1], dim=0)
    
    def init_state(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.latent_size),
               torch.zeros(self.num_layers, batch_size, self.latent_size))

class ExperienceReplay(Dataset):
    def __init__(self, alph, context_len=8, max_size = 1028):
        super(ExperienceReplay, self).__init__()
        self.data = list()
        self.alph = alph
        self.alph_len = len(alph) 
        self.ctx_len = context_len
        self.max_size = max_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.tokenize(self.data[idx])

    def get_batch(self, bs):
        idxs = torch.randint(0, self.__len__(), (bs,))
        batch = torch.stack([self.data[idx] for idx in idxs], dim = 0)
        return self.tokenize(batch[:,:-1]), batch[:,1:]

    def insert(self, context):
        self.data.append(torch.LongTensor([self.alph[c] for c in context]))

        if self.__len__() >= self.max_size:
            idxs = torch.randint(0,self.max_size//2,(self.max_size//4,))
            for idx in idxs:
                self.forget(idx)

    def forget(self, idx):
        self.data.pop(idx)

    def tokenize(self, x):
        return F.one_hot(x, self.alph_len)

