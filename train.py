import torch
from torch import nn

from config import Config
from data import dataloader
from model import Transformer

if __name__ == '__main__':

    transformer = Transformer(Config).to(Config.device)
    transformer.train()

    loss_fn = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)

    for epoch in range(Config.epochs):
        loss_sum = 0
        for pad_enc_input, pad_dec_input in dataloader:
            pad_enc_input = pad_enc_input.to(Config.device)
            pad_dec_input = pad_dec_input.to(Config.device)
            dec_pred = transformer(pad_enc_input, pad_dec_input)
            dec_input_gt = pad_dec_input.to(Config.device)

            loss = loss_fn(dec_pred.reshape(-1,
                                            dec_pred.size()[-1]),
                           dec_input_gt.reshape(-1))
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'epoch:{epoch}, loss:{loss.item()}')
        torch.save(transformer, f'epoch_{epoch}.pth')
