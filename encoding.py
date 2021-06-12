from collections import deque
MAX= 0xffff
Q1 = 0x4000 
HALF= 0x8000 
Q3= 0xc000 

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
        d = self.high - self.low + 1
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
                c = 2*c + self.out_buffer.popleft()
            dat.append(c)
        return dat

class Decoder:
    def __init__(self, alph_len):
        self.low = 0 
        self.high = MAX
        self.code = 0
        self.alph_len = alph_len
        self.buffer = deque()

    def __len__(self):
        return self.buffer.__len__()

    def update(self, prob, cum_prob): 
        if len(self.buffer) == 0:
            raise ValueError("Bit buffer is empty")

        d = self.high - self.low +1
        self.low += int(cum_prob*d)
        self.high = self.low + int(prob*d)
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= Q1 and self.high < Q3:
                self.code -= Q1
                self.low -= Q1
                self.high -= Q1
            else:
                break
            self.low *= 2
            self.high = 2*self.high +1
            self.code = 2*self.code + self.buffer.popleft()
    
    def decode(self, probs):
        i = 0
        cum_prob = 0
        d = self.high - self.low +1
        cp = (self.code - self.low +1)/d
        while cum_prob + probs[i].item() < cp:
            cum_prob += probs[i].item() 
            i += 1
        prob = probs[i].item()
        #if prob < 1/self.alph_len:
        #    prob = 1/self.alph_len
        #    cum_prob = 0
        #    i = 0
        #    while cum_prob + prob < cp:
        #        cum_prob += prob
        #        i += 1
        
        return i, prob, cum_prob

    def init_code(self):
        self.code = 0
        for _ in range(16):
            self.code = 2*self.code +self.buffer.popleft()

    def insert_byte(self, b):
        c = int.from_bytes(b, byteorder="big")
        for j in reversed(range(8)):
            self.buffer.append( (c>>j)&1)
