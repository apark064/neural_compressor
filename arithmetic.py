from nnet import *
import torch
import codecs
import os
import sys
from torch import nn
from torch.nn import functional as F
from collections import deque, defaultdict

MAX= 0xffff
Q1 = 0x4000 
HALF= 0x8000 
Q3= 0xc000 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

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
    bs = kwargs.get("bs",32)
    epochs = kwargs.get("epochs",1)
    
    model.train()
    for e in range(epochs):
        state = model.init_state(bs, device)
        inps, targs = data.get_batch(bs)
        
        inps, targs = inps.to(device), targs.to(device)
        for s in state:
            s= s.to(device)
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
        dat = '' 
        while len(self.out_buffer) >= 8:
            c = 0 
            for _ in range(8):
                c <<= 1
                c += self.out_buffer.popleft()
            dat += chr(c) 
        return dat

class Decoder:
    def __init__(self, in_stream):
        self.low = 0 
        self.high = MAX
        self.code = 0

        self.in_stream = in_stream
        self.buffer = deque()
        self.out_buffer = deque()

    def update(self, prob, cum_prob): 
        d = self.high - self.low +1
        self.lower += int(cum_prob*d)
        self.upper = self.lower + int(cum_prob*d)
        while True:
            if self.low >= HALF:
                self.low -= HALF
                self.upper -= HALF
                self.code -= HALF
            elif self.lower >= Q1 and self.upper < Q3:
                self.code -= Q1
                self.low -= Q1
                self.upper -= Q1
            else:
                break
        self.lower *= 2
        self.upper = 2*self.upper +1
        self.code = 2*self.code + self.buffer.pop()

        if len(self.buffer) == 0:
            c = int.from_bytes(self.in_stream.read(1), "little")
            for _ in range(8):
                buffer.append( c &1)
                c >>= 1

def LSTM_decode_file(file_name, **kwargs):
    ctx_len = kwargs.get("ctx_len", 64)
    n_layers = kwargs.get("n_layers",2)
    lr = kwargs.get("lr", 5e-3)

    f = codecs.open(file_name, 'r')
    orig_name, n_bytes = f.readline().strip('\n').split()
    n_bytes = int(n_bytes)
    alph = { c: idx for idx, c in enumerate(f.readline().strip('\n'))}
    alph_len = len(alph)
    const_prob = 1/alph_len

    loss = nn.CrossEntropyLoss()
    replay_data = ExperienceReplay(alph, ctx_len)
    encoder = Encoder(alph_len)
    model = CharLSTM(280, alph_len, num_layers = n_layers)
    ctx = deque( f.readline())
    

def LSTM_encode_file(file_name, **kwargs):
    ctx_len = kwargs.get("ctx_len",8)
    lr = kwargs.get("lr", 5e-3)
    out_file = kwargs.get("out_file", "ballsinmy.face")
    n_layers = kwargs.get("n_layers",3)

    freq = file_stats(file_name)
    alph_len = len(freq)
    alph = { c : idx for idx, c in enumerate(freq.keys()) } 
    n_bytes = sum(freq.values())
    const_prob = 1/alph_len
 
    model = CharLSTM(150, alph_len, num_layers = n_layers)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    loss = nn.CrossEntropyLoss()
    replay_data = ExperienceReplay(alph, ctx_len)
    encoder = Encoder(alph_len)

    stream = codecs.open(file_name, mode='r', encoding='utf-8')
    output = codecs.open(out_file, mode='w')
    ctx = deque( [alph[c] for c in stream.read(ctx_len)] )
    for c in ctx:
        replay_data.insert(c)


    '''
    FILE_NAME LENGTH
    ALPHABET
    INITIAL CONTEXT
    '''
    output.write(file_name + " " + str(n_bytes) + '\n')
    for c in alph.keys():
        output.write(c)
    output.write('\n')
    for c in ctx:
        output.write(str(c))
    output.write('\n')

    count = 0 

    next_chr = stream.read(1)
    for i in range(n_bytes):
        replay_data.insert(alph[next_chr])
        with torch.no_grad():
            probs = model.predict(torch.LongTensor(list(ctx)), device = device)
            prob = probs[ alph[next_chr] ].item()
            cum_prob = probs[:alph[next_chr]].sum().item()

        if prob < const_prob:
            prob = const_prob
            cum_prob = const_prob*alph[next_chr] 

        print(f"{prob:.07f} {count}\{i}")
        encoder.encode_char(prob, cum_prob)
        if len(encoder) > 64:
            count += 8 
            encoder.flush_buffer() 
            #output.write(encoder.flush_buffer())

        ctx.popleft()
        ctx.append(alph[next_chr])
        next_chr = stream.read(1)
        
        if (i-1)%100 == 0:
            train(model, replay_data, optimizer, loss, epochs = 15)

    output.close()
    stream.close()
