#!/usr/bin/python3
import nnet
from arithmetic_coder import *
import torch
import argparse
import os
import sys
from torch import nn
from torch.utils.data import DataLoader 
from torch.nn import functional as F

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ascii_one_hot(n):
    t = torch.LongTensor([int.from_bytes(n, 'little')])
    return F.one_hot(t, 128).float()

def train(model, optimizer, loss_function, file_name, batch_size = 32): 
    file_size = os.path.getsize(file_name)
    f = open(file_name, 'rb') 

    state = torch.zeros(1,1,model.latent_size)
    
    next_char = f.read(1)
    states = []
    chars = []
    nexts= []

    for i in range(1,file_size):
        
        x = ascii_one_hot(next_char)
        states.append(state.data)
        chars.append(x.data)
        out, state = model(x.unsqueeze(0), state)
        next_char = f.read(1)
        nexts.append(int.from_bytes( next_char, 'little'))
        
        if i%batch_size == 0:
            latents = torch.cat(states, dim=1)
            inputs = torch.cat(chars, dim=0).unsqueeze(1)

            targets = torch.LongTensor(nexts)
            outputs, _ = model(inputs,latents)
            loss = loss_function(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            probs = F.softmax(outputs[0], dim=0)
            print(probs[targets[0]].item())

            states  = []
            chars = []
            nexts = []

    f.close()

if __name__ == "__main__":
    torch.manual_seed(420)

    model = nnet.CharPredictor(64, 128)
    model.train()
    err = nn.CrossEntropyLoss() 
    optim = torch.optim.AdamW(model.parameters(), lr = 0.01, amsgrad = True)
    train(model, optim, err, sys.argv[1])
    
    print("beginning")
    with torch.no_grad():
        model.eval()
        num_bytes = 0
        encoder = Encoder(model)
        f = open(sys.argv[1], 'rb')
        next_char = f.read(1)

        #while next_char != b'':
        for _ in range(5000):
            encoder.encode_char(next_char)
            next_char = f.read(1)

            if len(encoder) > 64:
                out = encoder.output_bits()
                num_bytes += 1
    #print(num_bytes)



