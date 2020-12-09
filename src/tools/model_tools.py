import torch
from torch import nn

def inference(model, images=None, loader=None):
    label = None
    if images is None:
        with torch.no_grad():
            for batch in loader:
                x1, x2, label = batch
                if torch.cuda.is_available():
                    x1, x2 = x1.to('cuda'), x2.to('cuda')
                model.eval()
                model.freeze()
                yield x1.cpu(), x2.cpu(), label, nn.Sigmoid()(model(x1, x2)).cpu()

    else:
        x1, x2 = zip(*images)
        with torch.no_grad():
            model.eval()
            model.freeze()
            return x1, x2, label, nn.Sigmoid()(model(x1, x2)).cpu()

