''' 
Encoder:
    * train the nnet on file
    * encode the file with arithmetic encoder
    * send the weights (prob large af) and compressed file
Decoder:
    * load nnet with the weights
    * decompress

''' 
from collections import defaultdict
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os

class CharPredictor(nn.Module):
    def __init__(self, latent_size, alph_size):
        super(CharPredictor, self).__init__()
        self.latent_size = latent_size
        self.alph_size = alph_size
        self.rnn = nn.LSTM(input_size = alph_size,
                          hidden_size = latent_size,
                          batch_first = True)
        self.lin = nn.Linear(latent_size,
                             alph_size,
                             bias = False)

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

class TextData(Dataset):
    def __init__(self, file_name, seq_len):
        super(TextData, self).__init__()
        assert os.path.exists(file_name)

        self.alph = defaultdict(int)
        with open(file_name, 'r') as file:
            dat = file.read()
            self.file_size = len(dat)
            for char in dat:
                self.alph[char] += 1

        self.alph_size = len(self.alph)

        # order characters by frequency
        reorder = sorted(self.alph.keys(),
                        key = lambda k: -self.alph[k])
        self.token = {reorder[i] : i for i in range(len(reorder))}
        self.inv_token = {i: reorder[i] for i in range(len(reorder))}

        seqs = []
        for i in range(seq_len, self.file_size, seq_len):
            seqs.append(
                self.tokenize(dat[i-seq_len:i])
            )
        self.seqs = torch.stack(seqs)

    def tokenize(self, x):
        return torch.tensor(
            list( map(lambda k: self.token[k], x))
        )

    def __getitem__(self,idx):
        return F.one_hot(self.seqs[idx][:-1], self.alph_size).float(),\
                self.seqs[idx][1:]

    def __len__(self):
        return self.seqs.size(0)
