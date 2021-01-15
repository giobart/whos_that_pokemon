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
def calc_acc_sim_at_k(T, k, distance, indices, thr=0.95):
    """
    T : [nb_samples] (target labels)
    k : number of neighbors
    distance : 2-d array of distances between images by index
    thr : threshold under which the distance must be
    """
    res = 0
    for image, neighbors in enumerate( indices[:, :k] ):
        for neighbor in neighbors:
            if distance[image, neighbor] < thr and T[image] == T[neighbor]:
                res += 1
                break
    return res
    # res = sum( [1 for image, neighbors in enumerate( indices[:, :k] ) for mario in neighbors if dist[mario] < thr and lbls[mario] == lbls[image]] )
    # s = sum([1 for img_idx, closest_idx in enumerate(indices[:, 0]) if distance[img_idx, closest_idx]<thr])
    # return s / (1. * len(indices))

def calc_f_measure(T, distance, indices, beta=1.0, thr=0.95):
    """
    beta : beta value for calculating F-measure, can be adjusted
    """
    tp, fp, fn = 0, 0, 0
    for image, neighbor in enumerate( indices[:, 0] ):
        if distance[image, neighbor] < thr and T[image] == T[neighbor]:
            tp += 1
        elif distance[image, neighbor] > thr and T[image] == T[neighbor]:
            fn += 1
        elif distance[image, neighbor] < thr and T[image] != T[neighbor]:
            fp += 1
    tp *= (1 + beta + beta)
    fn *= (beta * beta)
    return ( tp / (tp + fp + fn) )
    
def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """
    Y = torch.from_numpy(Y)
    s = sum([1 for t, y in zip(T, Y) if t in y[:k]])
    return s / (1. * len(T))