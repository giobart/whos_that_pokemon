import torch
import numpy as np
from torch import nn
import sklearn
from src.tools import evaluation_tool

def inference(model, images=None, loader=None):
    if images is None:
        with torch.no_grad():
            for batch in loader:
                x1, x2, label = batch
                if torch.cuda.is_available():
                    x1, x2, model = x1.to('cuda'), x2.to('cuda'), model.to('cuda')
                model.eval()
                model.freeze()
                distance = model(x1, x2).squeeze()
                if isinstance(model.loss_fn, nn.BCEWithLogitsLoss):
                    print("BCE loss detected")
                    y_hat = nn.Sigmoid()(distance)
                    y_hat[y_hat >= 0.5] = 1.0
                    y_hat[y_hat < 0.5] = 0.0
                    distance = y_hat
                model.cpu()
                yield x1.cpu(), x2.cpu(), label, distance.cpu()

    else:
        x1, x2, label = zip(*images)

        with torch.no_grad():
            x1 = torch.stack([x for x in x1])
            x2 = torch.stack([x for x in x2])
            label = torch.stack([x for x in label])
            if torch.cuda.is_available():
                x1, x2, model = x1.to('cuda'), x2.to('cuda'), model.to('cuda')
            model.eval()
            model.freeze()
            distance = model(x1, x2)
            if isinstance(model.loss_fn, nn.BCEWithLogitsLoss):
                print("BCE loss detected")
                y_hat = nn.Sigmoid()(distance)
                y_hat[y_hat >= 0.5] = 1.0
                y_hat[y_hat < 0.5] = 0.0
                distance = y_hat

            model.cpu()
            yield x1.cpu(), x2.cpu(), label, distance.squeeze().cpu()


def inference_one(model, images=None, loader=None):
    """
    takes dataloader of list of tuples of images and labels and outputs the embedding (logits)
    :param model: LightningModule to use
    :param images: List of tuples: [(image1, label1), (image2, label2), ...]
    :param loader:
    :return:
    """
    if images is None:
        with torch.no_grad():
            for batch in loader:
                x1, label = batch
                if torch.cuda.is_available():
                    x1, model = x1.to('cuda'), model.to('cuda')

                model.eval()
                model.freeze()
                logits = model.forward_one(x1).squeeze()
                model.cpu()
                yield x1.cpu(), label, logits.cpu()
    else:
        x1, label = zip(*images)

        with torch.no_grad():
            x1 = torch.stack([x for x in x1])
            label = torch.stack([x for x in label])
            if torch.cuda.is_available():
                x1, model = x1.to('cuda'), model.to('cuda')

            model.eval()
            model.freeze()
            logits = model.forward_one(x1)
            model.cpu()
            yield x1.cpu(), label, logits.cpu()


def get_k_similar_group(model, images=None, loader=None, k=1):
    """
    Takes a model trained with the group loss. Outputs the indices and the distances of the most similar pairs
    :param model: LightningModule trained with group loss
    :param images: List of images: [image1, image2, ...]
    :param loader:
    :return: (x, y, indices, distances)
    """
    if images is None:
        for batch in loader:
            x, y = batch
            indices, distances = get_similar_ind(k, model=model, batch=batch)
            yield x, y, indices, distances
    else:
        x = torch.stack([img for img in images])
        indices, distances = get_similar_ind(k, model=model, images=x)
        yield x, None, indices, distances


def get_similar_ind(k, model=None, emb=None, batch=None, images=None):
    if emb is None:
        emb, _ = evaluation_tool.predict_batchwise(model, batch=batch, images=images)
    # rank the nearest neighbors for each input
    distances = sklearn.metrics.pairwise.pairwise_distances(emb)
    # get nearest points
    indices = np.argsort(distances, axis=1)[:, 1: k + 1]

    return indices, distances

def get_labeled_and_unlabeled_points(labels, num_points_per_class, num_classes=100):
    labs, L, U = [], [], []
    labs_buffer = np.zeros(num_classes)
    num_points = labels.shape[0]
    for i in range(num_points):
        if labs_buffer[labels[i]] == num_points_per_class:
            U.append(i)
        else:
            L.append(i)
            labs.append(labels[i])
            labs_buffer[labels[i]] += 1
    return labs, L, U

def assign_by_euclidian_at_k_indices(X, k):
    """
        X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
        k : for each sample, assign target labels of k nearest points
    """
    distances = sklearn.metrics.pairwise.pairwise_distances(X)
    # get nearest points
    indices = np.argsort(distances, axis=1)[:, 1: k + 1]
    return indices, distances

# def assign_by_euclidian_at_k(X, T, k):
#     """
#     X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
#     k : for each sample, assign target labels of k nearest points
#     """
#     # get nearest points
#     # indices, _ = assign_by_euclidian_at_k_indices(X, k)
#     indices, _ = get_similar_ind(k, emb=X)
#     return np.array([[T[i] for i in ii] for ii in indices], dtype=np.float64)

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance, label):
        loss_contrastive = torch.mean((1 - label) * torch.pow(distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        return loss_contrastive

