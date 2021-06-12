from collections import defaultdict
import codecs
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.distributions import dirichlet
import torch.nn.functional as F
import os

class CharLSTM(nn.Module):
    def __init__(self, latent_size, alph_len, **kwargs):
        super(CharLSTM, self).__init__()
        self.embed_dim = kwargs.get("embed_dim", 100)
        self.num_layers = kwargs.get("num_layers",2)
        self.latent_size = latent_size
        self.alph_len = alph_len
        self.embed = nn.Embedding(self.alph_len, self.embed_dim) 
        self.rnn = nn.LSTM(input_size = self.embed_dim,
                          hidden_size = latent_size,
                          num_layers = self.num_layers,
                          batch_first = True)
        self.lin = nn.Linear(latent_size, alph_len, bias = False)

        for n in range(self.num_layers):
            weights = [f"weight_ih_l{n}", f"weight_hh_l{n}"]
            biases = [f"bias_ih_l{n}", f"bias_hh_l{n}"]
            for w in weights:
                nn.init.orthogonal_(self.rnn.state_dict()[w])
            for b in biases:
                nn.init.zeros_(self.rnn.state_dict()[b])
                nn.init.ones_(self.rnn.state_dict()[b][self.latent_size:self.latent_size*2])

    def forward(self, x, state):
        out, state = self.rnn(self.embed(x), state)
        out = self.lin(out.flatten(end_dim =1))
        return out, state

    def predict(self, char, state = None, device = "cpu"):
        if state is None:
            state = self.init_state(device = device)
        inp = torch.LongTensor([char]).unsqueeze(0)
        inp = inp.to(device)
        for s in state:
            s = s.to(device)
        
        out, state = self.forward(inp, state)
        probs = F.softmax(out.squeeze(0), dim = 0) 
        return probs, state
    
    def init_state(self, batch_size=1, device = "cpu"):
        return (torch.randn(self.num_layers, batch_size, self.latent_size, device = device),
               torch.randn(self.num_layers, batch_size, self.latent_size, device = device))

class ExperienceReplay(Dataset):
    def __init__(self, alph, ctx_len=8, max_size = 2048):
        super(ExperienceReplay, self).__init__()
        self.data = list()
        self.alph = alph 
        self.alph_len = len(alph) 
        self.ctx_len = ctx_len
        self.max_size = max_size

    def __len__(self):
        return self.data.__len__() - self.ctx_len 

    def __getitem__(self,idx):
        return torch.LongTensor(self.data[idx: idx+self.ctx_len])

    def get_batch(self, bs):
        idxs = torch.randint(0, self.__len__(), (bs,))
        batch = torch.stack(
                [torch.LongTensor(self.data[idx: idx+self.ctx_len]) for idx in idxs], 
                dim = 0)
        return batch[:,:-1], batch[:,1:]

    def insert(self, char):
        self.data.append(char)
        
        #forget past data
        if( self.__len__() > self.max_size):
            self.data = self.data[self.__len__()//2 :]

