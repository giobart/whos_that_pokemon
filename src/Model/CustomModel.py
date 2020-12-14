import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from src.tools.model_tools import ContrastiveLoss
from src.Model.GeneralLayers import FCN_layer
from src.Model.CNN_Nets import myCNN, InceptionRNV1
from enum import Enum
from pytorch_lightning.metrics.functional import accuracy
from torch.optim.lr_scheduler import StepLR


class CNN_MODEL(Enum):
    MyCNN = 1
    InceptionResnetV1 = 2


class Siamese(pl.LightningModule):
    def __init__(self, hparams=None, scheduler_params=None, cnn_model=CNN_MODEL.InceptionResnetV1, freeze_layers=True):
        super().__init__()
        self.hparams = hparams
        self.loss_fn = hparams['loss_fn']
        self.scheduler_params = scheduler_params

        # CNN
        if cnn_model == CNN_MODEL.MyCNN:
            self.conv = myCNN(nn.PReLU,
                              hparams["filter_size"],
                              hparams["filter_channels"],
                              padding=int(hparams["filter_size"] / 2),
                              )  # size/8 x 4*channels
        elif cnn_model == CNN_MODEL.InceptionResnetV1:
            self.conv = InceptionRNV1(freeze_layers=freeze_layers)
        else:
            raise Exception("cnn_model is not defined correctly")

        self.input_size = self.conv.input_size
        self.cnn_output_size = self.conv.output_size

        if isinstance(self.loss_fn, ContrastiveLoss):
            self.loss_fn.margin = hparams['loss_margin']
            self.linear = nn.Sequential(
                FCN_layer(self.cnn_output_size, hparams['n_hidden1'], dropout=hparams['dropout']),
                FCN_layer(hparams['n_hidden1'], hparams['n_hidden2']),
                nn.Linear(hparams['n_hidden2'], hparams['n_hidden3'])
            )
        elif isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            self.linear = FCN_layer(self.cnn_output_size, hparams['n_hidden1'])
            self.out = nn.Sequential(
                # FCN_layer(hparams['n_hidden1'], 1, dropout=hparams['dropout']),
                FCN_layer(hparams['n_hidden1'], hparams['n_hidden2'], dropout=hparams['dropout']),
                nn.Linear(hparams['n_hidden2'], 1)
            )
        else:
            raise Exception('loss_fn is not well defined')

        self.params_to_update = []
        print("Layers to update")
        if freeze_layers:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.params_to_update.append(param)
                    print("\t", name)
        else:
            self.params_to_update = self.parameters()

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1).squeeze()
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        if isinstance(self.loss_fn, ContrastiveLoss):
            return F.pairwise_distance(out1, out2, keepdim=True)
        elif isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            return self.out(torch.abs(out1 - out2))
        else:
            raise Exception('loss_fn is not well defined')

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.params_to_update, lr=self.hparams["lr"],
                                     weight_decay=self.hparams["weight_decay"])

        if self.scheduler_params is not None:
            lr_scheduler = StepLR(optimizer, **self.scheduler_params)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def calc_acc(self, y, logits):
        y_hat = nn.Sigmoid()(logits).squeeze()
        y_hat[y_hat >= 0.5] = 1.0
        y_hat[y_hat < 0.5] = 0.0
        return accuracy(y_hat, y)
        # return torch.tensor(torch.sum(y == y_hat).item() / (len(y) * 1.0))

    def general_step(self, batch):
        x1, x2, y = batch
        output = self(x1, x2)

        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            loss = self.loss_fn(output.squeeze(), y)
            n_correct = self.calc_acc(y.detach().cpu(), output.detach().cpu())
        else:
            loss = self.loss_fn(output, y)
            n_correct = torch.tensor(0.0)

        return loss, n_correct

    def training_step(self, train_batch, batch_idx):
        self.train()
        loss, n_correct = self.general_step(train_batch)
        return {
            'loss': loss,
            'n_correct': n_correct
        }

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        loss, n_correct = self.general_step(val_batch)
        return {
            'loss': loss.detach().cpu(),
            'n_correct': n_correct
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def general_epoch_end(self, outputs, mode):  ### checked
        # average over all batches aggregated during one epoch
        logger = False if mode == 'test' else True
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['n_correct'] for x in outputs]).mean()
        self.log(f'{mode}_loss', avg_loss, logger=logger)
        self.log(f'{mode}_acc', avg_acc, logger=logger)
        return avg_loss, avg_acc

    def training_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        avg_loss, avg_acc = self.general_epoch_end(outputs, 'test')
        return {
            'avg_loss': avg_loss,
            'avg_acc': avg_acc
        }
