import nnet
import torch
from torch import nn
from torch.utils.data import DataLoader 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = CharPredictor(64, 128)
    model.train()
    err = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr = 0.01)


def train(model, optimizer, file_name, epochs):
    data = nnet.TextData(file_name, 8)
    loader = DataLoader(data, 
                              shuffle = False, 
                              pin_memory = False, 
                              batch_size = 32)
    for e in range(epochs):
        for i, data in enumerate(loader):
            inputs, states, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)





