from encoding import Encoder, Decoder 
import torch
import logging
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from collections import deque, defaultdict

class CharLSTM(nn.Module):
    def __init__(self, latent_size, alph_len, **kwargs):
        super(CharLSTM, self).__init__()
        self.embed_dim = kwargs.get("embed_dim", 100)
        self.num_layers = kwargs.get("num_layers",3)
        self.eps = kwargs.get("eps", 1e-3)
        self.latent_size = latent_size
        self.alph_len = alph_len
        self.embed = nn.Embedding(self.alph_len, self.embed_dim) 
        self.lstm= nn.LSTM(input_size = self.embed_dim,
                          hidden_size = latent_size,
                          num_layers = self.num_layers,
                          batch_first = True)
        self.lin = nn.Linear(latent_size, alph_len, bias = False)
        self.const_prob = 1/self.alph_len * torch.ones(self.alph_len)

        for n in range(self.num_layers):
            weights = [f"weight_ih_l{n}", f"weight_hh_l{n}"]
            biases = [f"bias_ih_l{n}", f"bias_hh_l{n}"]
            for w in weights:
                nn.init.orthogonal_(self.lstm.state_dict()[w])
            for b in biases:
                nn.init.zeros_(self.lstm.state_dict()[b])
                nn.init.ones_(self.lstm.state_dict()[b][self.latent_size:self.latent_size*2])

    def forward(self, x, state):
        out, state = self.lstm(self.embed(x), state)
        out = self.lin(out.flatten(end_dim =1))
        return out, state

    def predict(self, char, state = None, device = torch.device("cpu")):
        if state is None:
            state = self.init_state(device = device)
        inp = torch.LongTensor([char]).unsqueeze(0)
        inp = inp.to(device)
        for s in state:
            s = s.to(device)
        self.const_prob.to(device)
        
        out, state = self.forward(inp, state)
        probs = F.softmax(out.squeeze(0), dim = 0) 
        probs = (1-self.eps)*probs + self.eps*self.const_prob 
        return probs, state
    
    def init_state(self, batch_size=1, device = torch.device("cpu")):
        return (torch.randn(self.num_layers, batch_size, self.latent_size, device = device),
               torch.randn(self.num_layers, batch_size, self.latent_size, device = device))

class XPReplay(Dataset):
    def __init__(self, alph, ctx_len=8, max_size = 1024):
        super(XPReplay, self).__init__()
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

def train(model, data, optimizer, err, **kwargs):
    bs = kwargs.get("bs",16)
    epochs = kwargs.get("epochs",1)
    device = kwargs.get("device", torch.device("cpu"))

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
    ctx_len = kwargs.get("ctx_len",64)
    lr = kwargs.get("lr", 5e-3)
    out_file = kwargs.get("out","test.zop")
    n_layers = kwargs.get("n_layers", 3)
    device = kwargs.get("device", torch.device("cpu"))
    bs = kwargs.get("bs", 16)
    lvl = kwargs.get("lvl", 1)

    set_seed()
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
        output = open(out_file, 'w', encoding='utf-8')
        output.write(init_ctx)
        decoder = Decoder(alph_len)

        #load encoded bytes 
        for _ in range(16):
            decoder.insert_byte(stream.read(1))
        decoder.init_code()
    else:
        raise ValueError(f'Invalid mode') 
        
    logging.debug(f"ctx: {init_ctx}")
    logging.debug(f"alph len: {alph_len}")

    # initialize LSTM 
    model = CharLSTM( alph_len*3, alph_len, 
            embed_dim = alph_len*3//2,
            num_layers = n_layers)
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    loss = nn.CrossEntropyLoss()
    replay_data = XPReplay(alph, ctx_len)
    ctx = deque( [alph[c] for c in init_ctx]) 
    for c in ctx:
        replay_data.insert(c)

    count = 0 
    for i in range(n_bytes - ctx_len):
        if i%1000 == 0:
            state = model.init_state(device = device)
        with torch.no_grad():
            probs, state = model.predict(ctx[-1], state, device = device)

        if mode == "encode":
            next_chr = stream.read(1)
            prob = probs[ alph[next_chr] ].item()
            cum_prob = probs[:alph[next_chr]].sum().item()

            logging.debug(f"{prob:.06f}")
            encoder.encode_char(prob, cum_prob)
            if len(encoder) > 256:
                count += 32 
                data = encoder.flush_buffer()
                output.write(data)

        elif mode == "decode":
            idx, prob, cum_prob = decoder.decode(probs)
            logging.debug(f"{prob:.05f} {cum_prob:.05f}")

            if len(decoder) < 128:
                decoder.fill_buffer(stream, 16)
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
                    epochs = 1 + 4*lvl,
                    device = device)

    output.close()
    stream.close()
