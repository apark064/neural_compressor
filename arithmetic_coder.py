import nnet
import os
import torch
from torch import nn
from torch.nn import functional as F
from collections import deque

max_val = 0xfff 
q1 = 0x400 
half = 0x800 
q3 = 0xc00 

def ascii_one_hot(n):
    t = torch.LongTensor([int.from_bytes(n, 'little')])
    return F.one_hot(t, 128).float()

def train(model, optimizer, loss_function, file_name, batch_size = 32): 
    file_size = os.path.getsize(file_name)
    f = open(file_name, 'rb') 

    state = torch.zeros(1,1,32)
    
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
        
            #print(loss.data.item())
            probs = F.softmax(outputs[0], dim=0)
            #print(probs[targets[0]])

            states  = []
            chars = []
            nexts = []

    f.close()

class Encoder:
    def __init__(self, model):
        self.low = 0
        self.high = 0xfff
        self.n_bits = 0
        self.buffer = deque()
        self.model = model
        self.state = torch.zeros(1,1, self.model.latent_size) 
        self.probs = None

    def __len__(self):
        return self.buffer.__len__()

    def update(self, symbol):
        x, self.state = self.model(ascii_one_hot(symbol).unsqueeze(0), self.state)
        
        self.probs = F.softmax(x, dim=1).squeeze(0)


    def output_bits(self):
        while len(self.buffer)>= 8:
            out = 0 
            for _ in range(8):
                out *= 2 
                out += self.buffer.popleft()
            yield out

    def write_bits(self, bit):
        self.buffer.append(bit)
        self.buffer.extend(self.n_bits*[not bit])
        self.n_bits = 0
        
    def encode_char(self, symbol):
        range = self.high - self.low
        self.update(symbol)

        idx = int.from_bytes(symbol, 'little')
        cum_prob = self.probs[:idx].sum()

        print(self.probs[idx].item())
        self.low += range * cum_prob.item()
        self.high = self.low + range * self.probs[idx].item()

        while True:
            if (self.high < half):
                self.write_bits(0)
            elif (self.low >= half):
                self.write_bits(1)
                self.low -= half
                self.high -= half
            elif (self.low >= q1 and self.high < q3):
                self.n_bits += 1
                self.low -= q1
                self.high -= q1
            else:
                break
            
            self.low *= 2
            self.high = 2*self.high + 1


class Decoder:
    def __init__(self, model, file_name):
        self.low = 0
        self.high = 0xfff
        self.n_bits
        self.buffer = list()
        self.model = model
        self.probs = None

    def load_bits(self):
        if (len(self.buffer) < 16):
            print("Buffer not filled")
        for i in range(len(self.buffer)):
            n_bit = 2 * n_bit + self.buffer(0)
            self.buffer.pop(0)

    def decode(self, cum_prob):
        range = self.high - self.low
        self.high = self.low + range * cum_prob
        low = cum_prob * range

        while True:
            if (self.high < half):
                print("f")
            elif (self.low >= half):
                self.low -= half
                self.high -= half
            elif (self.low >= q1 and self.high < q3):
                self.low -= q1
                self.high -= q1
            else: 
                break
            self.low *= 2
            self.high = 2 * self.high + 1
            