max_val = 0xfff 
q1 = 0x400 
half = 0x800 
q3 = 0xc00 

class Encoder:
    def __init__(self, model, file_name):
        self.low = 0
        self.high = 0xfff
        self.n_bits = 0
        self.buffer = list()

    def cum_prob(self, symbol):
        pass

    def write_bits(self, bit):
        self.buffer.extend([bit] + self.n_bits*[not bit])
        self.n_bits = 0
        
    def encode_char(self, symbol, cum_prob):
        range = self.high - self.low
        high = self.low + range * cum_prob[symbol-1]
        low = self.low + range * cum_prob[symbol]

        while True:
            if (high < half):
                self.write_bits(0)
            elif (low >= half):
                self.write_bits(1)
                low -= half
                high -= half
            elif (low >= q1 and high < q3):
                n_bits += 1
                low -= q1
                high -= q1
            else:
                break
            
            low = 2*low
            high = 2*high + 1


class Decoder:
    def __init__(self):
        self.low = 0
        self.high = 0xfff
        self.n_bits
        self.buffer = list()

    def load_bits(self):
        if (len(self.buffer) < 16):
            print("Buffer not filled")
        for i in range(len(self.buffer)):
            n_bit = 2 * n_bit + self.buffer(0)
            self.buffer.pop(0)

    def decode(self, cum_prob):
        range = self.high - self.low
        high = self.low + range * cum_prob
        low = cum_prob * range

        while True:
            if (high < half):
                print("f")
            elif (low >= half):
                low -= half
                high -= half
            elif (low >= q1 and high < q3):
                low -= q1
                high -= q1
            else: 
                break
            low *= 2
            high = 2 * high + 1