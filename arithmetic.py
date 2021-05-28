from nnet import *
import torch
import codecs
import os
import sys
from torch import nn
from torch.nn import functional as F
from collections import deque, defaultdict

MAX= 0xfff
Q1 = 0x400 
HALF= 0x800 
Q3= 0xc00 

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def bit_stream(f_name):
    f = open(f_name, 'rb')
    byte = b'0' 
    while byte != b'':
        b = f.read(1)
        n = int.from_bytes(b, 'big')
        for i in range(b):
            yield (n >> i) & 1
    f.close()

def train(model, data, optimizer, err, **kwargs):
    bs = kwargs.get("bs",16)
    epochs = kwargs.get("epochs",1)
    
    model.train()
    for e in range(epochs):
        state = model.init_state(bs)
        inps, targs = data.get_batch(bs)
        
        inps, targs = inps.to(device), targs.to(device)
        out, state = model(inps, state)
        loss = err(out, targs.flatten())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    model.eval()
        
def shift_left(queue, x):
    queue.popleft()
    queue.append(x)
    
def file_stats(file_name):
    stats = defaultdict(int)
    with codecs.open(file_name, mode='r', encoding='utf-8') as f:
        c = f.read(1)
        while c != '':
            stats[c] += 1
            c = f.read(1)
    return stats

class Encoder:
    def __init__(self, alph_len = 256):
        self.low = 0  
        self.high = MAX
        self.n_bits = 0
        self.alph_len = alph_len
        self.out_buffer = deque()
    
    def __len__(self):
        return self.out_buffer.__len__()
    
    def encode_char(self, prob, cum_prob):
        d = self.high - self.low 
        self.low += int(d * cum_prob)
        self.high = self.low + int(d * prob)

        while True:
            assert self.high <= MAX and self.low >= 0
            if self.high < HALF:
                self.write_bits(0)
                self.n_bits = 0
            elif self.low >= HALF:
                self.write_bits(1)
                self.n_bits = 0
                self.low -= HALF
                self.high -= HALF
            elif self.low >= Q1 and self.high < Q3:
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
            self.out_buffer.append(not bit)
            self.n_bits -= 1

    def flush_buffer(self):
        dat = bytearray()
        while len(self.out_buffer) >= 8:
            c = 0 
            for _ in range(8):
                c <<= 1
                c += self.out_buffer.popleft()
            dat.append(c)
        return dat

class Decoder:
    def __init__(self, model):
        self.model = model
        self.low = 0 
        self.high = MAX
        self.code = 0
        self.buffer = deque()

    def decode_char(self): 
        pass

def encode_file(file_name, **kwargs):
    ctx_len = kwargs.get("ctx_len",8)
    lr = kwargs.get("lr", 5e-3)
    out_file = kwargs.get("out_file", "ballsinmyface.zip")
    
    freq = file_stats(file_name)
    alph_len = len(freq)
    alph = { c : idx for idx, c in enumerate(freq.keys()) } 

    n_bytes = sum(freq.values())
    freq_est = [ freq[c]/n_bytes for c in freq]

    model = CharLSTM(256, alph_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    loss = nn.CrossEntropyLoss()

    replay_data = ExperienceReplay(alph, ctx_len)
    encoder = Encoder(alph_len)

    stream = codecs.open(file_name, mode='r', encoding='utf-8')
    output = codecs.open(out_file, mode='wb')
    
    ctx = deque( list(stream.read(ctx_len)) )
    
    count = 0 

    for _ in range(8):
        count += 1
        replay_data.insert(ctx)
        next_chr  = stream.read(1)
        shift_left(ctx, next_chr)


    for i in range(5000):
        replay_data.insert(ctx)
        with torch.no_grad():
            probs = model.predict(replay_data[-1])
            prob = probs[ alph[next_chr] ].item()
            cum_prob = probs[:alph[next_chr]].sum().item()

        if prob < freq_est[alph[next_chr]]:
            prob = freq_est[alph[next_chr]]
            cum_prob = sum(freq_est[:alph[next_chr]])

        print(f"{prob:.07f} {count}")
        encoder.encode_char(prob, cum_prob)
        if len(encoder) > 64:
            count += 8 
            output.write(encoder.flush_buffer())
        
        ctx.popleft()
        ctx.append(next_chr)
        next_chr = stream.read(1)
        
        train(model, replay_data, optimizer, loss, epochs = min(i+1,10))

    output.close()
    stream.close()
