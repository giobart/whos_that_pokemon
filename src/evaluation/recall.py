import torch
import numpy as np
import sklearn.metrics.pairwise
from src.tools import model_tools

def assign_by_euclidian_at_k(X, T, k):
    """
    X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
    k : for each sample, assign target labels of k nearest points
    """
    # get nearest points
    indices, _ = model_tools.get_similar_ind(k, emb=X)
    return np.array([[T[i] for i in ii] for ii in indices], dtype=np.float64)

# def assign_by_euclidian_at_k_indices(X, k):
#     """
#         X : [nb_samples x nb_features], e.g. 100 x 64 (embeddings)
#         k : for each sample, assign target labels of k nearest points
#     """
#     distances = sklearn.metrics.pairwise.pairwise_distances(X)
#     # get nearest points
#     indices = np.argsort(distances, axis=1)[:, 1: k + 1]
#     return indices, distances

# TODO: calc acc similar to calc_recall but add restriction on the distance.
def calc_acc_sim_at_k(distance, indices, thr=0.95):
    pass
    # s = sum([1 for img_idx, closest_idx in enumerate(indices[:, 0]) if distance[img_idx, closest_idx]<thr])
    # return s / (1. * len(indices))

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    Y = torch.from_numpy(Y)
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))