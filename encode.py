max_val = 0xfff 
q1 = 0x400 
half = 0x800 
q3 = 0xc00 

def encode(symbol, cum_prob):
    low = 0
    high = max_val

    range = high - low
    high = low + range * cum_prob[symbol-1]
    low = low + range * cum_prob[symbol]

    while True:
        if (high < half):
            bit = 0
        elif (low >= half):
            bit = 1
            low -= half
            high -= half
        elif (low >= q1 and high < q3):
            bit += 1
            low -= q1
            high -= q1
        else:
            break
        low = 2*low
        high = 2*high + 1

