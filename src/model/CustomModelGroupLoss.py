import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from src.tools.model_tools import get_labeled_and_unlabeled_points
from src.model.CNN_Nets import BnInception
from enum import Enum
from torch.optim.lr_scheduler import StepLR
from src.tools import gtg
import sys
from src.tools.evaluation_tool import GroupRecall

class CNN_MODEL_GROUP(Enum):
    MyCNN = 1
    BN_INCEPTION = 2

class Siamese_Group(pl.LightningModule):
    """This is not really a Siamese Network but can be considered as a generalization of it. It trains BnInception Model
    with the Group Loss."""
    def __init__(self, hparams={}, scheduler_params={}, nb_classes=10177, finetune=False,
                 cnn_state_dict=None, calc_train_stats=False):
        """
        :param hparams: Optimizer parameters
        :param scheduler_params: Scheduler parameters
        :param nb_classes: Number of classes to train on
        :param finetune: whether to finetune the model (train on classification task) or to train using the group loss
        :param cnn_state_dict: used to init the model weights
        :param calc_train_stats: whether to calc stats during training
        """
        super().__init__()
        self.hparams = hparams
        self.scheduler_params = scheduler_params
        self.gtg = gtg.GTG(nb_classes, max_iter=1, device='cuda')
        self.criterion = nn.NLLLoss()
        self.criterion2 = nn.CrossEntropyLoss()
        self.scaling_loss = 1.0
        self.nb_classes = nb_classes
        self.finetune = finetune
        self.recall_metric = GroupRecall()
        self.calc_train_stats = calc_train_stats


        self.model = BnInception(num_classes=self.nb_classes)

        if not finetune and cnn_state_dict is not None:
            self.model.load_state_dict(cnn_state_dict)

        self.input_size = self.model.input_size
        self.params_to_update = self.parameters()

    def forward_one(self, x):
        x, fc = self.model(x)
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
        is_training = self.model.training
        recall = self.recall_metric(fc, batch)
        # recall, _ = evaluation_tool.evaluate(self, fc7=fc, batch=batch)
        if is_training:
            self.train()
        # return torch.tensor(recall)
        return recall

    def general_step(self, batch, mode):
        x, y = batch
        logits, fc = self(x)
        labs, L, U = get_labeled_and_unlabeled_points(labels=y,
                                                       num_points_per_class=self.hparams['num_labeled_points_class'],
                                                       num_classes=self.nb_classes)

        probs_for_gtg = F.softmax(logits / self.hparams['temperature'], dim=-1)

        # do GTG (iterative process)
        probs_for_gtg, W = self.gtg(fc, fc.shape[0], labs, L, U, probs_for_gtg)
        probs_for_gtg = torch.log(probs_for_gtg + 1e-12)

        nll = self.criterion(probs_for_gtg, y)
        ce = self.criterion2(logits, y)
        loss = self.scaling_loss * nll + ce
        self.test_loss_nan(loss)
        recall = torch.tensor([0])
        if mode == 'val' or (mode == 'train' and self.calc_train_stats):
            recall = self.calc_recall(batch, fc)

        return loss, nll, ce, recall

    def general_step_finetune(self, batch):
        x, y = batch
        logits, fc = self(x)

        loss = self.criterion2(logits, y)
        self.test_loss_nan(loss)

        return loss

    def test_loss_nan(self, loss):
        # check possible net divergence
        if torch.isnan(loss):
            print("We have NaN numbers, closing")
            print("\n\n\n")
            sys.exit(0)

    def training_step(self, train_batch, batch_idx):
        self.train()

        if self.finetune:
            loss = self.general_step_finetune(train_batch)
            return {
                'loss': loss,
            }
        else:
            loss, nll, ce, recall = self.general_step(train_batch, mode='train')
            return {
                'loss': loss,
                'recall': recall,
                'nll': nll,
                'ce': ce,
            }

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        if self.finetune:
            loss = self.general_step_finetune(val_batch)
            return {
                'loss': loss,
            }
        else:
            loss, nll, ce, recall = self.general_step(val_batch, mode='val')
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


        self.log(f'{mode}_loss', avg_loss, logger=logger)
        self.log(f'{mode}_nll_loss', avg_nll, logger=logger)
        self.log(f'{mode}_ce_loss', avg_ce, logger=logger)


        avg_recall = torch.stack([x['recall'] for x in outputs]).mean(dim=0)
        which_nearest_neighbors = [1]
        for i, k in enumerate(which_nearest_neighbors):
            self.log(f'{mode}_R%_@{k}', 100 * avg_recall[i], logger=logger)

        return avg_loss.cpu(), avg_recall.cpu(), avg_nll.cpu(), avg_ce.cpu()

    def general_epoch_end_finetune(self, outputs, mode):  ### checked
        # average over all batches aggregated during one epoch
        logger = False if mode == 'test' else True
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log(f'{mode}_loss', avg_loss.cpu(), logger=logger)

        return avg_loss

    def training_epoch_end(self, outputs):
        if self.finetune:
            self.general_epoch_end_finetune(outputs, 'train')
        else:
            self.general_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        if self.finetune:
            self.general_epoch_end_finetune(outputs, 'val')
        else:
            self.general_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        if self.finetune:
            avg_loss, avg_recall, avg_nll, avg_ce = self.general_epoch_end_finetune(outputs, 'test')
            return {
                'avg_loss': avg_loss.cpu(),
            }
        else:
            print('running test')
            avg_loss, avg_recall, avg_nll, avg_ce = self.general_epoch_end(outputs, 'test')

            return {
                'avg_loss': avg_loss.cpu(),
                'avg_recall@1': avg_recall[0].cpu(),
                'avg_nll_loss': avg_nll.cpu(),
                'avg_ce_loss': avg_ce.cpu(),
            }