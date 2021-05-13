import os
import torch
from torch import nn
from torch.nn import functional as F 
from torch.utils.data import Dataset

''' 
Encoder:
    * train the nnet on file
    * encode the file with arithmetic encoder
    * send the weights (prob large af) and compressed file
Decoder:
    * load nnet with the weights
    * decompress

''' 

class CharPredictor(nn.Module):
    def __init__(self, latent_size, alph_size):
        super(CharPredictor, self).__init__()
        self.latent_size = latent_size
        self.alph_size = alph_size
        self.gru = nn.GRU(input_size = alph_size,
                          hidden_size = latent_size,
                          batch_first = True)    
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p = 0.4)
        self.lin = nn.Linear(latent_size, alph_size, bias = False)

    def forward(self, x, state = None):
        if state is None:
            state = torch.zeros(x.size(0), x.size(1), self.latent_size)
        out, state = self.gru(x, state)
        out = self.lin(out.view(-1, self.latent_size))
        return out, state

class TextData(Dataset):
    def __init__(self, file_name, context_len):
        super(TextData, self).__init__()
        self.context_len = context_len
        assert os.path.exists(file_name)
        file_size = os.path.get_size(file_name)
        self.size = file_size - context_len
        self.data = torch.Tensor(self.size, context_len)
        self.target = torch.Tensor(self.size, 1)
        
        with open(file_name, 'rb') as file:
            byte_arr = file.read(file_size)

        for i in range(self.size):
            for j in range(context_len):
                self.data[i][j] = byte_arr[i+j]
            self.target[i] = byte_arr[i+context_len]
            
    def __getitem__(self,idx):
        return self.data[idx], self.target[idx]

    def __len__(self):
        return self.size