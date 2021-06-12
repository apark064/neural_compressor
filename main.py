from compress import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Neural Compressor")
    parser.add_argument('file', metavar='F', type = str, nargs = 1)
    parser.add_argument("-o", dest = 'o', type = str, help = "output file name")
    parser.add_argument("-c", action = "store_true") 
    parser.add_argument("-d", action = "store_true")
    args = parser.parse_args()

    LSTM_compress(args.file[0], 
            mode = "decode" if args.d else "encode", 
            out = args.o)
