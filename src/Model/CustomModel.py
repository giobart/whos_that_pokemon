import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from src.tools.model_tools import ContrastiveLoss


class ConvPoolPRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activ_fn=nn.PReLU):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(2, 2),
            activ_fn(),
            nn.BatchNorm2d(out_channels),
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

class FCN_layer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.PReLU(),
            nn.BatchNorm1d(output_size),
            nn.Dropout(p=dropout),
        )

        def weights_init(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight, gain=2.0)

        self.model.apply(weights_init)

    def forward(self, x):
        return self.model(x)

class Siamese(pl.LightningModule):
    def __init__(self, hparams=None, input_shape=(3, 128, 128), CNN_model=CNN):
        super().__init__()
        self.hparams = hparams
        input_channels, height, width = input_shape
        # CNN
        self.conv = CNN_model(nn.PReLU,
                              hparams["filter_size"],
                              hparams["filter_channels"],
                              padding=int(hparams["filter_size"] / 2),
                              input_channels=input_channels)  # size/8 x 4*channels

        self.loss_fn = hparams['loss_fn']
        if isinstance(self.loss_fn, ContrastiveLoss):
            self.linear = nn.Sequential(
                FCN_layer(int(hparams["filter_channels"] * 4 * int(height / 8) * int(width / 8)),
                          hparams['n_hidden1'],
                          dropout=hparams['dropout']),
                FCN_layer(hparams['n_hidden1'], hparams['n_hidden2']),
                nn.Linear(hparams['n_hidden2'], hparams['n_hidden3'])
            )
        elif isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            self.loss_fn.margin = hparams['loss_margin']
            self.linear = FCN_layer(int(hparams["filter_channels"] * 4 * int(height / 8) * int(width / 8)),
                                    hparams['n_hidden1'])
            self.out = nn.Sequential(
                FCN_layer(hparams['n_hidden1'], hparams['n_hidden2'], dropout=hparams['dropout']),
                nn.Linear(hparams['n_hidden2'], 1)
            )
        else:
            raise Exception('loss_fn is not well defined')

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])
        return optimizer

    def calc_acc(self, y, output, beta=1.0, boundary=0.995):
        if isinstance(self.loss_fn, ContrastiveLoss):
            pred = (output > boundary )
            # return torch.tensor(torch.sum(y == pred).item() / (len(y) * 1.0))
            tp = ( 1 + beta * beta ) * torch.sum( y * pred == 1 ).item()
            fn = ( beta * beta ) * torch.sum( y > pred ).item()
            fp = torch.sum( y < pred ).item()
            return torch.tensor( tp / ( tp + fn + fp ) )
        else:
            y_hat = nn.Sigmoid()(output)
            pred = (y_hat > 0.5)
            return torch.tensor(torch.sum(y == pred).item() / (len(y) * 1.0))

    def general_step(self, batch):
        x1, x2, y = batch
        output = self(x1, x2)
        loss = self.loss_fn(output, y)
        n_correct = torch.tensor(0.0)
        if isinstance(self.loss_fn, ContrastiveLoss):
            n_correct = self.calc_acc(y.detach().cpu(), output.detach().cpu() )
        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            n_correct = self.calc_acc(y.detach().cpu(), output.detach().cpu() )

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