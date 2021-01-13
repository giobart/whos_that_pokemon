import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from src.model.CNN_Nets import BnInception
from torch.optim.lr_scheduler import StepLR
import sys
from .CustomModelGroupLoss import Siamese_Group
from src.model.inception_bn import bn_inception
from pytorch_lightning.metrics.functional import accuracy
from src.tools import evaluation_tool
from src.tools.evaluation_tool import GroupRecall

class Classification_Trainer(pl.LightningModule):
    def __init__(self, hparams={}, scheduler_params={}, nb_classes=1, pretrained_classes=-1, checkpoint_path=''):

        super().__init__()
        self.hparams = hparams
        self.scheduler_params = scheduler_params
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaling_loss = 1.0
        self.nb_classes = nb_classes


        if pretrained_classes != -1:
            pre_model = Siamese_Group(nb_classes=pretrained_classes)
            pre_model = pre_model.load_from_checkpoint(checkpoint_path=checkpoint_path)
            state_dict = pre_model.model.state_dict()
            for old_key in list(state_dict.keys()):
                new_key = '.'.join(old_key.split('.')[1:])
                state_dict[new_key] = state_dict.pop(old_key)

            self.model = bn_inception(pretrained=False)
            self.model.last_linear = nn.Linear(1024, pretrained_classes)
            self.model.load_state_dict(state_dict, strict=True)
            self.model.requires_grad_(False)
            self.model.last_linear = nn.Linear(1024, self.nb_classes).requires_grad_(True)

            self.input_size = pre_model.input_size
        else:
            self.model = bn_inception(pretrained=False)
            self.model.last_linear = nn.Linear(1024, self.nb_classes)
            self.input_size = 224
        # if not finetune and (weights_path is not None or cnn_state_dict is not None):
        #     self.model.load_state_dict(cnn_state_dict)


        self._show_params_to_update()

    def _show_params_to_update(self):
        self.params_to_update = []
        print("Layers to update")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params_to_update.append(param)
                print("\t", name)
        else:
            self.params_to_update = self.parameters()

    def forward_one(self, x):
        x, fc = self.model(x)
        return x.squeeze(), fc

    def forward(self, x):
        x, fc = self.forward_one(x)
        return x.squeeze(), fc

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

    def calc_acc(self, logits, y):
        y_hat = nn.Sigmoid()(logits).squeeze()
        y_hat[y_hat >= 0.5] = 1.0
        y_hat[y_hat < 0.5] = 0.0
        return accuracy(y_hat, y)

    def general_step(self, batch, mode):
        x, y = batch
        logits, fc = self(x)
        loss = self.criterion(logits, y.float())
        self.test_loss_nan(loss)
        return loss, self.calc_acc(logits, y)

    def test_loss_nan(self, loss):
        # check possible net divergence
        if torch.isnan(loss):
            print("We have NaN numbers, closing")
            print("\n\n\n")
            sys.exit(0)

    def training_step(self, train_batch, batch_idx):
        self.train()
        loss, acc = self.general_step(train_batch, mode='train')
        return {
            'loss': loss,
            'acc': acc,
        }

    def validation_step(self, val_batch, batch_idx):
        self.eval()
        loss, acc = self.general_step(val_batch, mode='val')
        return {
            'loss': loss,
            'acc': acc,
        }

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def general_epoch_end(self, outputs, mode):  ### checked
        # average over all batches aggregated during one epoch
        logger = False if mode == 'test' else True
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        self.log(f'{mode}_loss', avg_loss, logger=logger)
        self.log(f'{mode}_acc', avg_acc, logger=logger)

        return avg_loss.cpu(), avg_acc.cpu()

    def training_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        avg_loss, avg_acc = self.general_epoch_end(outputs, 'test')
        return {
            'avg_loss': avg_loss.cpu(),
            'acc': avg_acc.cpu(),
        }

    def infer(self, x):
        if torch.cuda.is_available():
            x = x.to('cuda')

        is_training = self.training
        self.eval()
        logits, _ = self(x)
        y_hat = nn.Sigmoid()(logits).squeeze()
        if is_training:
            self.train()

        return y_hat.cpu().detach()