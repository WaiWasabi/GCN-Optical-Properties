import numpy as np
from model.emax_regression import GCN
from model.train_step import train_cro_only
import torch


def grid_search(loader, lr_range, lr_increment, embed_range, embed_increment, epochs, device):
    lrs = list(np.arange(*lr_range, lr_increment))
    embeds = list(np.arange(*embed_range, embed_increment))
    loss_arr = np.zeros([len(lrs), len(embeds)])
    log = [[0]*len(embeds)]*len(lrs)
    print(len(log)*len(log[0]))
    for x, lr in enumerate(lrs):
        for y, embed in enumerate(embeds):
            model = GCN(embed)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = torch.nn.MSELoss()
            losses = []
            for epoch in range(epochs):
                loss, embedding = train_cro_only(model, loader, loss_fn, optimizer, device)
                losses.append(loss)
            loss_arr[x][y] = losses[-1]
            log[x][y] = losses
            print(f"finished training with lr={lr}, embedding size={embed}")
    return log, loss_arr
