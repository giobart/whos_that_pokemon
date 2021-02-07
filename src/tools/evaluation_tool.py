import torch.nn.functional as F
from src import evaluation
import torch
import logging
from pytorch_lightning.metrics import Metric


def predict_batchwise(model, dataloader=None, fc7=None, batch=None, images=None, nb_batches=0):
    """
    :param model: Model used for the prediction
    :param dataloader: batch is None, uses the dataloader for the prediction
    :param fc7: model output before the classifcation layer. If not None, the model will not be used.
    :param batch: if not None, the batch will be used instead of the dataloader
    :param images: images as pytorch tensor. If provided they will be used as input of the model instead of the images
                    in the batch. Useful when inferencing and a batch is not available
    :param nb_batches: if >0 and dataloader is not None, limit the number of batches to be processed from the dataloader
    :return: Tuple of concatenated outputs and labels. If images are provided then labels returned will be None.
    """
    if model is not None:
        model.eval()

    fc7s, L = [], []
    with torch.no_grad():
        if batch is None and dataloader is not None:
            for i, batch in enumerate(dataloader):
                fc7_out, Y = inference_group(model, fc7, batch=batch)
                fc7s.append(fc7_out)
                L.append(Y)
                if nb_batches>0:
                    if i>nb_batches:
                        break

            return torch.cat(fc7s).squeeze(), torch.cat(L).squeeze()
        else:
            fc7, Y = inference_group(model, fc7, batch=batch, X=images)
            fc7s.append(fc7)
            L.append(Y)
            return torch.cat(fc7s).squeeze(), Y

def inference_group(model, fc7, batch=None, X=None):
    """

    :param model: Model used for the prediction
    :param fc7: model output before the classifcation layer. If not None, the model will not be used.
    :param batch: if not None, the batch will be used instead of the dataloader
    :param X: images as pytorch tensor. If provided they will be used as input of the model instead of the images
                    in the batch. Useful when inferencing and a batch is not available
    :return: Tuple of outputs and labels. If X is not None then labels returned will be None.
    """
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


def evaluate(model, dataloader=None, fc7=None, batch=None, calc_nmi=False, nb_batches=0):
    """

    :param model: Model used for the prediction
    :param dataloader: batch is None, uses the dataloader for the prediction
    :param fc7: model output before the classifcation layer. If not None, the model will not be used.
    :param batch: if not None, the batch will be used instead of the dataloader
    :param calc_nmi: whether to compute nmi or not
    :param nb_batches: if >0 and dataloader is not None, limit the number of batches to be processed from the dataloader
    :return: tuple containing a list of recall@1, 10, and 20, and nmi value if calc_nmi is True, else None.
    """
    nb_classes = model.nb_classes

    model_is_training = model.training
    model.eval()

    # calculate embeddings with model, also get labels
    emb, labels = predict_batchwise(model, dataloader=dataloader, fc7=fc7, batch=batch, nb_batches=nb_batches)

    nmi = None
    if dataloader is not None and calc_nmi:
        num_included_classes = len(torch.unique(labels))
        print("Calculating NMI for ", num_included_classes, "classes")
        nmi = evaluation.calc_normalized_mutual_information(labels, evaluation.cluster_by_kmeans(emb, num_included_classes))

    recall = []
    # rank the nearest neighbors for each input
    k_pred_labels = evaluation.assign_by_euclidian_at_k(emb, labels, 20)
    if batch is None:
        which_nearest_neighbors = [1, 10, 20]
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
    """Recall implementation as pytorch Metric"""
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