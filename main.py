from compress import LSTM_compress 
import argparse
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "NRL Compress")
    parser.add_argument('infile', nargs=1, type=str, help = "input file name")
    parser.add_argument("-d", action = "store_true", help = "decompress")
    parser.add_argument("-l", dest = "l", choices = {'1','2','3'}, default = 1, help = "training level")
    args = parser.parse_args()

    if not os.path.isfile(args.infile[0]):
        raise ValueError("Couldn't read file") 
    out_file = args.infile[0] + ".nrl" 

    LSTM_compress(args.infile[0], 
            mode = "decode" if args.d else "encode", 
            out = out_file,
            lvl = eval(args.l) )
