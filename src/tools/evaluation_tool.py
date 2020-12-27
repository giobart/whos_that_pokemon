import torch.nn as nn
import torch.nn.functional as F
from src import evaluation
import torch
import logging
import sys
import numpy as np


def predict_batchwise(model, dataloader=None, fc7=None, batch=None, images=None):
    net_type = model.CNN_MODEL

    fc7s, L = [], []
    with torch.no_grad():
        if batch is None:
            for batch in dataloader:
                fc7, Y = inference_group(model, batch, fc7)
                fc7s.append(fc7)
                L.append(Y)
        else:
            fc7, Y = inference_group(model, fc7, batch=batch, X=images)
            fc7s.append(fc7)
            L.append(Y)

        fc7, Y = torch.cat(fc7s), torch.cat(L)
        return torch.squeeze(fc7), torch.squeeze(Y)


def inference_group(model, fc7, batch=None, X=None):
    if X is None:
        X, Y = batch
    else:
        Y = None

    if fc7 is None:
        X = X.cuda() if torch.cuda.is_available() else X
        _, fc7 = model(X)

    # normalize the features in the unit ball
    fc7 = F.normalize(fc7, p=2, dim=1)
    if Y is None:
        return fc7.cpu(), None

    return fc7.cpu(), Y.cpu()


def evaluate(model, dataloader=None, fc7=None, batch=None):

    model_is_training = model.training
    model.eval()

    # calculate embeddings with model, also get labels
    emb, labels = predict_batchwise(model, dataloader=dataloader, fc7=fc7, batch=batch)
    recall = []
    # rank the nearest neighbors for each input
    k_pred_labels = evaluation.assign_by_euclidian_at_k(emb, labels, 1000)
    which_nearest_neighbors = [1, 10, 100, 1000]

    for k in which_nearest_neighbors:
        r_at_k = evaluation.calc_recall_at_k(labels, k_pred_labels, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

    model.train(model_is_training)  # revert to previous training state

    return recall
