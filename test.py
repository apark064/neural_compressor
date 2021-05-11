#!/usr/bin/python3

import nnet
import torch


model = nnet.CharPredictor(64, 128)
state = torch.zeros(1,1,64)
x = torch.ones(1,1,128)

out, _ = model(x, state)
print(out)
