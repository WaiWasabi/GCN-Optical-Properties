import torch


def train_cro_only(model, loader, loss_fn, optimizer, device, history=False):
    log = []
    loss, embedding = (None, None)
    for batch in loader:
        batch.to(device)
        optimizer.zero_grad()
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)

        loss = torch.sqrt(loss_fn(batch.y, pred))
        loss.backward()
        optimizer.step()
        log.append([pred, batch.y])
    if history:
        return loss, embedding, log
    return loss, embedding


def train_with_solvent(model, loader, loss_fn, optimizer, device, history=False):
    log = []
    loss, embedding = (None, None)
    for batch in loader:
        batch.to(device)
        optimizer.zero_grad()
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = torch.sqrt(loss_fn(batch.y, pred))
        loss.backward()
        optimizer.step()
        log.append([pred, batch.y])
    if history:
        return loss, embedding, log
    return loss, embedding
