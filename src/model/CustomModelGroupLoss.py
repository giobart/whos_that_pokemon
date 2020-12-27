import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from src.tools.model_tools import ContrastiveLoss, get_labeled_and_unlabeled_points
from src.model.GeneralLayers import FCN_layer
from src.model.CNN_Nets import myCNN, InceptionRNV1
from enum import Enum
from pytorch_lightning.metrics.functional import accuracy
from torch.optim.lr_scheduler import StepLR
from src.tools import gtg
import sys
from src.tools import evaluation_tool

class CNN_MODEL_GROUP(Enum):
    MyCNN = 1
    InceptionResnetV1 = 2

class Siamese_Group(pl.LightningModule):
    def __init__(self, hparams=None, scheduler_params=None, cnn_model=CNN_MODEL_GROUP.MyCNN, freeze_layers=True,
                 nb_classes=10177):

        super().__init__()
        self.hparams = hparams
        # self.loss_fn = hparams['loss_fn']
        self.scheduler_params = scheduler_params
        self.freeze_layers = freeze_layers
        self.gtg = gtg.GTG(nb_classes, max_iter=1, device='cuda')
        self.criterion = nn.NLLLoss()
        self.criterion2 = nn.CrossEntropyLoss()
        self.scaling_loss = 1.0
        self.temperature = hparams['temperature']
        self.num_labeled_points_class = hparams['num_labeled_points_class']
        self.nb_classes = nb_classes
        self.CNN_MODEL = cnn_model

        # CNN
        if cnn_model == CNN_MODEL_GROUP.MyCNN:
            self.conv = myCNN(nn.PReLU,
                              hparams["filter_size"],
                              hparams["filter_channels"],
                              padding=int(hparams["filter_size"] / 2),
                              )  # size/8 x 4*channels
        elif cnn_model == CNN_MODEL_GROUP.InceptionResnetV1:
            self.conv = InceptionRNV1(freeze_layers=freeze_layers)
        else:
            raise Exception("cnn_model is not defined correctly")

        self.input_size = self.conv.input_size
        self.cnn_output_size = self.conv.output_size

        self.linear = nn.Sequential(
            FCN_layer(self.cnn_output_size, hparams['n_hidden1'], dropout=hparams['dropout']),
            FCN_layer(hparams['n_hidden1'], hparams['n_hidden2']),
        )
        self.classifier = nn.Linear(hparams['n_hidden2'], nb_classes)

        self._show_params_to_update()

    def _show_params_to_update(self):
        self.params_to_update = []
        print("Layers to update")
        if self.freeze_layers:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.params_to_update.append(param)
                    print("\t", name)
        else:
            self.params_to_update = self.parameters()

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1).squeeze()
        fc = self.linear(x)
        x = self.classifier(fc)
        return x, fc

    def forward(self, x):
        x, fc = self.forward_one(x)
        return x, fc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.params_to_update, lr=self.hparams["lr"],
                                     weight_decay=self.hparams["weight_decay"])

        if self.scheduler_params is not None:
            lr_scheduler = {
                'scheduler': StepLR(optimizer, **self.scheduler_params)
            }

            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def calc_recall(self, batch, fc):
        recall, _ = evaluation_tool.evaluate(self, fc7=fc, batch=batch)
        return torch.tensor(recall)

    def general_step(self, batch):
        x, y = batch
        logits, fc = self(x)
        labs, L, U = get_labeled_and_unlabeled_points(labels=y,
                                                       num_points_per_class=self.num_labeled_points_class,
                                                       num_classes=self.nb_classes)

        probs_for_gtg = F.softmax(logits / self.temperature, dim=-1)

        # do GTG (iterative process)
        probs_for_gtg, W = self.gtg(fc, fc.shape[0], labs, L, U, probs_for_gtg)
        probs_for_gtg = torch.log(probs_for_gtg + 1e-12)

        nll = self.criterion(probs_for_gtg, y)
        ce = self.criterion2(logits, y)
        loss = self.scaling_loss * nll + ce
        self.test_loss_nan(loss)

        recall = self.calc_recall(batch, fc)
        return loss, nll, ce, recall

    def test_loss_nan(self, loss):
        # check possible net divergence
        if torch.isnan(loss):
            print("We have NaN numbers, closing")
            print("\n\n\n")
            sys.exit(0)

    def training_step(self, train_batch, batch_idx):
        self.train()
        loss, nll, ce, recall = self.general_step(train_batch)
        return {
            'loss': loss,
            'recall': recall,
            'nll': nll,
            'ce': ce,
        }

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        loss, nll, ce, recall = self.general_step(val_batch)
        return {
            'loss': loss,
            'recall': recall,
            'nll': nll,
            'ce': ce,
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def general_epoch_end(self, outputs, mode):  ### checked
        # average over all batches aggregated during one epoch
        logger = False if mode == 'test' else True
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_nll = torch.stack([x['nll'] for x in outputs]).mean()
        avg_ce = torch.stack([x['ce'] for x in outputs]).mean()

        avg_recall = torch.stack([x['recall'] for x in outputs]).mean(dim=0)

        self.log(f'{mode}_loss', avg_loss, logger=logger)
        self.log(f'{mode}_nll_loss', avg_nll, logger=logger)
        self.log(f'{mode}_ce_loss', avg_ce, logger=logger)

        which_nearest_neighbors = [1]
        for i, k in enumerate(which_nearest_neighbors):
            self.log(f'{mode}_R%_@{k} : ', 100 * avg_recall[i], logger=logger)

        return avg_loss, avg_recall, avg_nll, avg_ce

    def training_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        avg_loss, avg_recall, avg_nll, avg_ce = self.general_epoch_end(outputs, 'test')
        return {
            'avg_loss': avg_loss,
            'avg_recall@1': avg_recall[0],
            'avg_nll_loss': avg_nll,
            'avg_ce_loss': avg_ce
        }