from nnet import *
import torch
import codecs
import argparse
import os
import sys
from torch import nn
from torch.utils.data import DataLoader 
from torch.nn import functional as F
from collections import deque

MAX= 0xfff
Q1 = 0x400 
HALF= 0x800 
Q3= 0xc00 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bit_stream(f_name):
    f = open(f_name, 'rb')
    byte = b'0' 
    while byte != b'':
        b = f.read(1)
        n = int.from_bytes(b, 'big')
        for i in range(j):
            yield (n >> i) & 1
    f.close()


def train(model, dataset, epochs, **kwargs):
    lr = kwargs.get("lr", 1e-2)
    optim = torch.optim.AdamW(net.parameters(), lr = lr)
    err = nn.CrossEntropyLoss()
    model.train()
    model.to(device)
    
    losses = []
    for e in range(epochs):
        print(f"EPOCH {e+1}")
        state = torch.zeros((1,1,net.latent_size), device = device)
        for i in range(len(dataset)):
            in_seq, targs = dataset[i]
            for s in state:
                s = s.to(device)
                s.detach_()
            
            in_seq, targs = in_seq.to(device), targs.to(device)
            loss = 0
            out, state = model(in_seq.unsqueeze(0), state)
            loss = err(out, targs).mean()

            losses.append(loss.item())
            i += 1
            if(i % 99 == 0):
                print( sum(losses)/100 )
                losses = []
            
            optim.zero_grad()
            loss.backward()
            optim.step()

class Encoder:
    def __init__(self, alph_len):
        self.low = 0  
        self.high = MAX
        self.n_bits = 0
        self.alph_len = alph_len
        self.out_buffer = deque()
    
    def encode_char(self, next_char, prob, cum_prob):
        probs = model.predict(next_char)
        cum_prob = model.cum_prob(next_char)

        d = high - low + 1 
        low += int(d * cum_prob)
        high = low + int(d * prob)

        while True:
            if high < HALF:
                self.write_bits(0)
                self.n_bits = 0
            elif low >= HALF:
                self.write_bits(1)
                self.n_bits = 0
                self.low -= HALF
                self.high -= HALF
            elif low >= Q1 and high < Q3:
                self.n_bits += 1
                self.low -=Q1
                self.high -= Q1
            else:
                break
            self.low *= 2
            self.high = 2*self.high + 1

    def write_bits(self, bit):
        self.out_buffer.append(bit)
        while self.n_bits > 0:
            self.out_buffer.append(!bit)
            self.n_bits -= 1

    def flush_buffer(self, n_bytes = 0):
        while len(self.out_buffer) >= 8:
            c = 0
            for _ in range(8):
                c += self.out_buffer.popleft()
                c <<= 1
            yield c


class Decoder:
    def __init__(self, model):
        self.model = model
        self.low = 0 
        self.high = MAX
        self.code = 0
        self.buffer = deque()

    def decode_char(self): 
        pass


def encode_file(file_name, encoder, **kwargs):
    ctx_len = kwargs.get("ctx_len",8)
    alph_len = kwargs.get("alph_len", 256)
    in_stream = codecs.open(file_name, mode='r', encoding='utf-8')
    replay_data = ExperienceReplay(ctx_len, alph_len)
    ctx = 
    replay_data.insert( 

