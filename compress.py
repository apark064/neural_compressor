from nnet import *
from encoding import * 
import torch
import os
import sys
import logging
from torch import nn
from torch.nn import functional as F
from collections import deque, defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
def shift_left(queue, x):
    queue.popleft()
    queue.append(x)
    
def file_stats(file_name):
    stats = defaultdict(int)
    with open(file_name, mode='r', encoding='utf-8') as f:
        c = f.read(1)
        while c != '':
            stats[c] += 1
            c = f.read(1)
    return stats

def set_seed(n = 163):
    torch.manual_seed(n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def LSTM_compress(file_name, mode = "encode", **kwargs):
    ctx_len = kwargs.get("ctx_len",32)
    lr = kwargs.get("lr", 5e-3)
    out_file = kwargs.get("out","test.zop")
    n_layers = kwargs.get("n_layers", 2)
    bs = kwargs.get("bs", 16)

    set_seed()
    #logging.basicConfig(filename="test.log", level=logging.DEBUG, format = '%(message)s')
    logging.basicConfig(level = logging.DEBUG)

    if mode == "encode":
        freq = file_stats(file_name)
        alph_len = len(freq)
        alph = { c : idx for idx, c in enumerate(freq.keys()) } 
        n_bytes = sum(freq.values())
        encoder = Encoder(alph_len)

        header = '{}\n{}\n{}{}'
        stream = open(file_name, 'r', encoding='utf-8')
        init_ctx = stream.read(ctx_len)
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(header.format( 
                    n_bytes, alph_len, ''.join(alph.keys()), init_ctx))
        output = open(out_file, 'ab+')

    elif mode == "decode":
        stream = open(file_name, 'rb')
        n_bytes = int(stream.readline().decode('utf-8').strip('\n'))
        alph_len = int(stream.readline().decode('utf-8').strip('\n'))
        alph = stream.read(alph_len).decode('utf-8')
        alph = { c: idx for idx, c in enumerate(alph)}
        idx_to_chr = { idx: c for idx, c in enumerate(alph)}
        init_ctx = stream.read(ctx_len).decode('utf-8')

        logging.debug(f"ctx: {init_ctx}")
        output = open(out_file, 'w', encoding='utf-8')
        output.write(init_ctx)
        decoder = Decoder(alph_len)

        #load encoded bytes 
        for _ in range(16):
            decoder.insert_byte(stream.read(1))
        decoder.init_code()
        
    # initialize LSTM 
    model = CharLSTM(300, alph_len, 
            embed_dim = 170,
            num_layers = n_layers)
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    loss = nn.CrossEntropyLoss()
    replay_data = ExperienceReplay(alph, ctx_len)
    ctx = deque( [alph[c] for c in init_ctx]) 
    for c in ctx:
        replay_data.insert(c)

    count = 0 

    for i in range(n_bytes - ctx_len):
        if i%1000 == 0:
            state = model.init_state()
        with torch.no_grad():
            probs, state = model.predict(ctx[-1], state)

        if mode == "encode":
            next_chr = stream.read(1)
            prob = probs[ alph[next_chr] ].item()
            cum_prob = probs[:alph[next_chr]].sum().item()
            if prob < 1/alph_len:
                prob = 1/alph_len 
                cum_prob = alph[next_chr]*prob

            #logging.debug(f"{prob:.07f} {count}\{i}")
            encoder.encode_char(prob, cum_prob)
            if len(encoder) > 256:
                count += 32 
                data = encoder.flush_buffer()
                logging.debug(f"wrote {data}")
                output.write(data)

        elif mode == "decode":
            idx, prob, cum_prob = decoder.decode(probs)

            #fill bit buffer
            if len(decoder) < 8:
                for _ in range(8):
                    byte = stream.read(1)
                    byte = '\x00' if byte == b'' else byte
                    decoder.insert_byte(byte)
            
            decoder.update(prob, cum_prob)
            next_chr = idx_to_chr[idx]
            output.write(next_chr)
            logging.debug(f"decoded: {next_chr}")

        ctx.popleft()
        ctx.append(alph[next_chr])
        replay_data.insert(alph[next_chr])
        
        if (i-1)%bs == 0:
            train(model, replay_data, optimizer, loss, 
                    bs = bs, 
                    epochs = 5)

    output.close()
    stream.close()
