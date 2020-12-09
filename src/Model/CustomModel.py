import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


class ConvPoolPRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activ_fn=nn.PReLU):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_channels),
            activ_fn()
        )

        def weights_init(m):
            if type(m) == nn.Conv2d:
                nn.init.xavier_normal_(m.weight, gain=2.0)

        self.model.apply(weights_init)

    def forward(self, x):
        return self.model(x)


class CNN(nn.Module):
    def __init__(self, activ_fn, f_size, f_channels, padding=0, input_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            ConvPoolPRelu(input_channels, f_channels, f_size, padding, activ_fn=activ_fn),  # 128x128xf
            ConvPoolPRelu(f_channels, f_channels * 2, f_size, padding, activ_fn=activ_fn), #64x64x2f
            ConvPoolPRelu(f_channels * 2, f_channels * 4, f_size, padding, activ_fn=activ_fn), #32x32x4f
            # ConvPoolPRelu(f_channels * 4, f_channels * 8, f_size, padding, activ_fn=activ_fn), #16x16x8f
        )

    def forward(self, x):
        return self.model(x)

class FCN(nn.Module):
    def __init__(self, hparams, input_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hparams["n_hidden1"]),
            nn.PReLU(),
            nn.BatchNorm1d(hparams["n_hidden1"]),
            # nn.Dropout(p=hparams["dropout"]),

            # nn.Linear(hparams["n_hidden1"], hparams["n_hidden2"]),
            # nn.ReLU(),
            # nn.BatchNorm1d(hparams["n_hidden2"]),
            # nn.Dropout(p=hparams["dropout"]),
        )


    def forward(self, x):
        return self.model(x)

class Siamese(pl.LightningModule):
    def __init__(self, hparams=None, input_channels=3, input_shape=(3, 128, 128), CNN_model=CNN):
        super().__init__()
        self.hparams = hparams

        # Parameters
        ## optimizer
        self.lr = hparams["lr"]
        self.weight_decay = hparams["weight_decay"]

        ## Model
        channels, height, width = input_shape
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.conv = CNN_model(nn.PReLU,
                              hparams["filter_size"],
                              hparams["filter_channels"],
                              padding=int(hparams["filter_size"] / 2),
                              input_channels=input_channels)  # size/8 x 4*channels

        self.linear = FCN(hparams, int(hparams["filter_channels"] * 4 * int(height / 8) * int(width / 8)))
        self.out = nn.Linear(hparams['n_hidden1'], 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1).squeeze()
        x = self.linear(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        diff = torch.abs(out1 - out2)
        out = self.out(diff)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def calc_acc(self, y, logits):
        y_hat = nn.Sigmoid()(logits)
        label = (y_hat > 0.5)
        return torch.tensor(torch.sum(y == label).item() / (len(y) * 1.0))

    def general_step(self, batch):
        x1, x2, y = batch
        logits = self(x1, x2)
        loss = self.loss_fn(logits, y)
        n_correct = self.calc_acc(y.detach().cpu(), logits.detach().cpu())
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
        self.log(f'{mode}_loss', torch.stack([x['loss'] for x in outputs]).mean())
        self.log(f'{mode}_acc', torch.stack([x['n_correct'] for x in outputs]).mean())

    def training_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.general_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self.general_epoch_end(outputs, 'test')