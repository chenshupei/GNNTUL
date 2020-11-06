import sys
import os
import torch
import random
import math

from sklearn.utils import shuffle
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

torch.set_default_tensor_type(torch.FloatTensor)


def apply_model2(train_loader, model):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    loss_fun = nn.MSELoss()
    loss_fun = nn.CrossEntropyLoss()
    correct = 0
    train_loss = 0
    total = 0
    correct5 = 0

    for step, (x, y) in enumerate(train_loader):
        x = x.to('cuda')
        y = y.to('cuda')
        out = model(x)
        y = y.float()
        _, y = y.max(1)
        loss = loss_fun(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (epoch + 1) % 100 == 0:  # 每 100 次输出结果
        #     print('Epoch: {}, Loss: {:.5f}'.format(epoch + 1, loss.data[0]))

        train_loss += loss.item()
        _, predicted = out.max(1)
        total += y.size(0)
        # _, y = y.max(1)
        correct += predicted.eq(y).sum().item()

        top_k = 5
        out_np = out.cpu().detach().numpy()
        # top_k_idx = out.detach().numpy().argsort(axis=1)[::-1][0:5]
        for index, o in enumerate(out_np):
            top5 = o.argsort()[::-1][:top_k]
            if y[index].item() in top5:
                correct5 = correct5 + 1

        if step % 50 == 0:
            print(step, len(train_loader), 'Loss: %.8f | Acc: %.3f%% (%d/%d) | Acc@5: %.3f%% (%d/%d)'
                  % (train_loss / (step + 1), 100. * correct / total, correct, total, 100. * correct5 / total, correct5, total))


def test_model2(model, test_loader):
    model.eval()
    correct = 0
    correct5 = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(torch.squeeze(inputs, 1))
            targets = targets.float()
            loss = 0

            test_loss += 0
            _, predicted = outputs.max(1)
            total += targets.size(0)
            _, targets = targets.max(1)
            correct += predicted.eq(targets).sum().item()

            top_k = 5
            out_np = outputs.cpu().detach().numpy()
            for index, o in enumerate(out_np):
                top5 = o.argsort()[::-1][:top_k]
                if targets[index].item() in top5:
                    correct5 = correct5 + 1

            targets = targets.cpu().numpy().tolist()
            predicted = predicted.cpu().numpy().tolist()

            y_true = y_true + targets
            y_pred = y_pred + predicted

            macro_f1 = f1_score(y_true, y_pred, average='macro')
            macro_P = precision_score(y_true, y_pred, average='macro')
            macro_R = recall_score(y_true, y_pred, average='macro')

            if batch_idx % 20 == 0:
                print(batch_idx, len(test_loader),
                      '******** Loss: %.8f | Acc: %.3f%% (%d/%d) | Acc@5: %.3f%% (%d/%d) | macro_f1: %.3f%% | macro_P: %.3f%% | macro_R: %.3f%%'
                      % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total, 100. * correct5 / total, correct5, total, 100. * macro_f1,
                         100. * macro_P, 100. * macro_R))