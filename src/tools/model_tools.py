import torch
import numpy as np
from torch import nn

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
