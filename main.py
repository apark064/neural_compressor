#!/usr/bin/env python3
from nnet import *
import torch
import argparse
import os
import sys
from torch import nn
from torch.utils.data import DataLoader 
from torch.nn import functional as F
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_val = 0xfff 
q1 = 0x400 
half = 0x800 
q3 = 0xc00 

def train(net, dataset, epochs, **kwargs):
    lr = kwargs.get("lr", 1e-2)
    
    optim = torch.optim.AdamW(net.parameters(), lr = lr, amsgrad = True)
    err = nn.CrossEntropyLoss()
    net.train()
    model.to(device)
    
    losses = []
    for e in range(epochs):
        state = ( torch.zeros((1,1,net.latent_size), device = device), 
                torch.zeros((1,1,net.latent_size), device = device))
        i = 0
        for in_seq, targs in dataset:
            for s in state:
                s = s.to(device)
                s.detach_()
            
            in_seq, targs = in_seq.to(device), targs.to(device)
            loss = 0
            out, state = model(in_seq.unsqueeze(0), state)
            loss = err(out, targs).mean()

            losses.append(loss.item())
            i += 1
            if(i%99 == 0):
                print( sum(losses)/100 )
                losses = []
            
            optim.zero_grad()
            loss.backward()
            optim.step()

def write_bits(bit, n_bits, buffer):
    buffer.append(bit)
    buffer.extend(n_bits*[not bit])

def encode(model, data):
    buffer = deque()
    n_bits = 0
    text = torch.flatten(data.seqs)

    low = 0
    high = 0xfff

    state = ( torch.zeros((1,1,model.latent_size), device = device), 
            torch.zeros((1,1,model.latent_size), device = device))

    n_bytes = 0
    for j in range(len(text)-1):
        d = high - low + 1 

        #print(data.inv_token[text[j].item()], end = '')
        x = F.one_hot(text[j], data.alph_size)
        x = x.to(device)
        probs, state = model(x.reshape(1,1,-1), state)
        probs = F.softmax(probs.squeeze(0), dim=0)
        cum_prob = probs[:text[j+1]].sum()

        #print(cum_prob.item())
        #print(probs[text[j+1]].item())

        low += int(d * cum_prob.item())
        high = low + int( d* probs[text[j+1]].item())

        while True:
            #print(f'low: {low:0b} \t\t high: {high:0b}')
            assert low >= 0 and high < 0x1000
            if (high < half):
                write_bits(False, n_bits, buffer)
                n_bits = 0
            elif (low >= half):
                write_bits(True, n_bits, buffer)
                n_bits = 0
                low -= half
                high -= half
            elif (low >= q1 and high < q3):
                n_bits += 1
                low -= q1
                high -= q1
            else:
                break
            
            low *= 2
            high = 2*high + 1
        
        while len(buffer) >= 8:
            byte = 0
            for _ in range(8):
                byte *= 2
                byte += int(buffer.popleft())
            n_bytes += 1
            
            if n_bytes%1000 == 0:
                print(n_bytes)
        
    print(n_bytes)
        

if __name__ == "__main__":
    dat = TextData(sys.argv[1],16)
    model = CharPredictor(256, dat.alph_size)
    train(model, dat, 20)
    #model.to(device)
    model.eval()
    with torch.no_grad():
        encode(model, dat)