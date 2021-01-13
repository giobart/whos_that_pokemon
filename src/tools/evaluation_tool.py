import torch.nn.functional as F
from src import evaluation
import torch
import logging
from pytorch_lightning.metrics import Metric

def predict_batchwise(model, dataloader=None, fc7=None, batch=None, images=None):
    if model is not None:
        model.eval()

    fc7s, L = [], []
    with torch.no_grad():
        if batch is None and dataloader is not None:
            for batch in dataloader:
                fc7_out, Y = inference_group(model, fc7, batch=batch)
                fc7s.append(fc7_out)
                L.append(Y)

            return torch.cat(fc7s).squeeze(), torch.cat(L).squeeze()
        else:
            fc7, Y = inference_group(model, fc7, batch=batch, X=images)
            fc7s.append(fc7)
            L.append(Y)
            return torch.cat(fc7s).squeeze(), Y

def inference_group(model, fc7, batch=None, X=None):
    if model is not None:
        model.eval()

    if X is None:
        X, Y = batch
    else:
        Y = None

    if fc7 is None:
        if torch.cuda.is_available():
            X = X.cuda()
            model = model.cuda()

        _, fc7 = model(X)

    # normalize the features in the unit ball
    fc7 = F.normalize(fc7, p=2, dim=1)

    if Y is None:
        return fc7.cpu(), None

    return fc7.cpu(), Y.cpu()


def evaluate(model, dataloader=None, fc7=None, batch=None, calc_nmi=False):
    nb_classes = model.nb_classes

    model_is_training = model.training
    model.eval()

    # calculate embeddings with model, also get labels
    emb, labels = predict_batchwise(model, dataloader=dataloader, fc7=fc7, batch=batch)

    nmi = None
    if dataloader is not None and calc_nmi:
        nmi = evaluation.calc_normalized_mutual_information(labels, evaluation.cluster_by_kmeans(emb, nb_classes))

    recall = []
    # rank the nearest neighbors for each input
    k_pred_labels = evaluation.assign_by_euclidian_at_k(emb, labels, 1000)
    if batch is None:
        which_nearest_neighbors = [1, 10, 100, 1000]
    else:
        which_nearest_neighbors = [1]

    for k in which_nearest_neighbors:
        r_at_k = evaluation.calc_recall_at_k(labels, k_pred_labels, k)
        recall.append(r_at_k)
        logging.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

    if model_is_training:
        model.train()  # revert to previous training state

    return recall, nmi


class GroupRecall(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("recall", default=torch.tensor(0.0))
        self.add_state("total", default=torch.tensor(0.0))

    def update(self, fc7, batch):
        emb, labels = predict_batchwise(None, fc7=fc7, batch=batch)
        k_pred_labels = evaluation.assign_by_euclidian_at_k(emb, labels, 1000)
        self.recall += torch.tensor(evaluation.calc_recall_at_k(labels, k_pred_labels, k=1))
        self.total += 1

    def compute(self):
        return torch.tensor([self.recall.float() / self.total])