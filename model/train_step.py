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


def train_with_solvent(model, loader, loss_fn, optimizer, device, history=False):  # for with_solvent.SolventGCN
    log = []
    loss, embedding = (None, None)
    for batch in loader:
        batch.to(device)
        print(batch)
        optimizer.zero_grad()
        pred, embedding = model(batch.x_c.float(), batch.edge_c, batch.attrib_c, batch.x_c_batch,
                                batch.x_s.float(), batch.edge_s, batch.attrib_s, batch.x_s_batch)
        loss = torch.sqrt(loss_fn(batch.y, pred))
        loss.backward()
        optimizer.step()
        log.append([pred, batch.y])
    if history:
        return loss, embedding, log
    return loss, embedding
