__all__ = ["train_loop"]

import torch
import torch.optim as torch_optim


def get_optimizer(model, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch_optim.AdamW(parameters, lr=lr)
    return optim


def train_model(model, optim, train_dl, loss_fn, use_gpu=False):
    model.train()
    total = 0
    sum_loss = 0
    for x1, x2, y in train_dl:
        if use_gpu:
            x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
        batch = y.shape[0]
        output = model(x1, x2)
        loss = loss_fn(output, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += batch
        sum_loss += batch * (loss.item())
    return sum_loss / total


def val_loss(model, valid_dl, loss_fn, use_gpu=False):
    model.eval()
    total = 0
    sum_loss = 0
    false_positives = 0
    true_positives = 0
    false_negatives = 0
    for x1, x2, y in valid_dl:
        if use_gpu:
            x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
        current_batch_size = y.shape[0]
        out = model(x1, x2)
        loss = loss_fn(out, y)
        sum_loss += current_batch_size * (loss.item())
        total += current_batch_size
        pred = torch.round(out)
        true_positives += (pred[y == 1] == 1).float().sum().item()
        false_positives += (pred[y == 0] == 1).float().sum().item()
        false_negatives += ((y == 1).float().sum().item()) - (pred[y == 1] == 1).float().sum().item()

    precision = true_positives / max(1, (true_positives + false_positives))
    recall = true_positives / max(1, (true_positives + false_negatives))
    print("valid loss %.3f, precision %.3f, recall %.3f" % (sum_loss / total, precision, recall))
    return sum_loss / total, precision, recall


def train_loop(model, epochs, loss_fn, train_dl, valid_dl, use_gpu=False, lr=0.01):
    max_f1 = 0
    max_params = {}
    optim = get_optimizer(model, lr=lr)
    for i in range(epochs):
        loss = train_model(model, optim, train_dl, loss_fn, use_gpu)
        print("training loss: ", loss)
        with torch.no_grad():
            loss, precision, recall = val_loss(model, valid_dl, loss_fn, use_gpu)
            f1 = precision * recall
            if f1 > max_f1:
                max_f1 = f1
                max_params = model.state_dict()
    return max_f1, max_params
