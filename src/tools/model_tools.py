import torch
import numpy as np
from torch import nn
from src.tools import evaluation_tool
import sklearn
from pytorch_lightning.metrics.functional import accuracy
import matplotlib.pyplot as plt
from src.modules.lfw_lightning_data_module import LfwImagesPairsDataset, LFW_DataModule
from src.tools.dataset_tools import get_dataset_filename_map, dataset_download_targz, get_pairs
import multiprocessing as mp

def inference(model, images=None, loader=None):
    """
    inference function used for the Siamese, which takes two images as input and outputs there embeddings.
    :param model: model used for inference
    :param images: if not None then they will be used instead of the images in the loader as input to the model.
    :param loader: dataloader with items (image1, image2, label)
    :return: generator of tuples (image1, image2, labe, distance)
    """
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
    :return: generator with tuples (image1, label, logits)
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
    """

    :param k: indices of the k most similar images to return
    :param model: model used to generate the indices
    :param emb: if embeddings are not None, no need to used the model to compute them
    :param batch: batch of (images, labels)
    :param images: images as pytorch tensor.
    :return:
    """
    with torch.no_grad():
        if emb is None:
            emb, _ = evaluation_tool.predict_batchwise(model, batch=batch, images=images)
        # rank the nearest neighbors for each input
        distances = sklearn.metrics.pairwise.pairwise_distances(emb)
        # get nearest points
        indices = np.argsort(distances, axis=1)[:, 1: k + 1]

        return indices, distances

def get_labeled_and_unlabeled_points(labels, num_points_per_class, num_classes=100):
    """Cridets to https://github.com/dvl-tum/group_loss"""
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
    Cridets to https://github.com/dvl-tum/group_loss
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
    Cridets to https://github.com/dvl-tum/group_loss
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

def eval_group_loss_lfw(model):
    dataset_download_targz()
    image_map = get_dataset_filename_map(min_val=1)
    pairs_map = get_pairs()
    train_dataset = LfwImagesPairsDataset(image_map, pairs_map["train"])
    val_dataset = LfwImagesPairsDataset(image_map, pairs_map["valid"])
    test_dataset = LfwImagesPairsDataset(image_map, pairs_map["test"])

    dataloader = LFW_DataModule(
        train_dataset,
        batch_size=32,
        splitting_points=None,
        num_workers=mp.cpu_count(),
        manual_split=True,
        valid_dataset=val_dataset,
        test_dataset=test_dataset,
        input_size=model.input_size
    )
    dataloader.setup()
    test_loader = dataloader.test_dataloader()


    init_bound = 0.85
    bounds = []
    accs = []

    while init_bound < 1:
        boundary = init_bound
        total_acc = 0

        for image1, image2, label in test_loader:
            images = torch.cat([image1, image2]).squeeze()
            with torch.no_grad():
                emb, _ = evaluation_tool.predict_batchwise(model, images=images)
                distances = sklearn.metrics.pairwise.pairwise_distances(emb)
                row_ind = list(range(int(len(distances) / 2)))
                col_ind = list(range(int(len(distances) / 2), len(distances)))

                distances_pairs = distances[row_ind, col_ind]
                distances_pairs[distances_pairs > boundary] = 1.0
                distances_pairs[distances_pairs < boundary] = 0.0
                y_hat = torch.tensor(distances_pairs)
                total_acc += accuracy(y_hat, label)

        total_acc = total_acc / len(test_loader)

        accs.append(total_acc)
        bounds.append(boundary)
        init_bound += 0.01

    plt.plot(bounds, accs)
    plt.title("Accuracy vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
