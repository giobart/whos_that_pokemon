import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


class ConvPoolPRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activ_fn = nn.PReLU):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.MaxPool2d(2, 2),  # 100
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
    def __init__(self, activ_fn, f_size, f_channels, padding=0, input_channels=3, input_shape=(3, 256, 256)):
        super().__init__()

        channels, height, width = input_shape

        self.model = nn.Sequential(
            ConvPoolPRelu(input_channels, f_channels, f_size, padding, activ_fn=activ_fn),
            ConvPoolPRelu(f_channels, f_channels * 2, f_size, padding, activ_fn=activ_fn),
            ConvPoolPRelu(f_channels * 2, f_channels * 4, f_size, padding, activ_fn=activ_fn),
            ConvPoolPRelu(f_channels * 4, f_channels * 8, f_size, padding, activ_fn=activ_fn),
            nn.Linear(int(8 * (height / 16) * (width / 16)), 4096),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)



class Siamese(pl.LightningModule):
    def __init__(self, hparams=None, input_channels=3, input_shape=(3, 256, 256), CNN_model=CNN):
        super().__init__()
        self.hparams = hparams

        ## Parameters
        f_channels = hparams["filter_channels"]
        f_size = hparams["filter_size"]
        self.lr = hparams["lr"]
        activ_fn = nn.PReLU
        padding = int(f_size/2)

        self.conv = CNN_model(activ_fn, f_size, f_channels, padding=padding, input_channels=input_channels, input_shape=input_shape), # size/16 x 8*channels
        self.out = nn.Linear(4096, 1)


    def forward(self, x1, x2):
        out1 = self.CNN_model(x1)
        out2 = self.CNN_model(x2)
        diff = torch.abs(out1 - out2)
        out = self.out(diff)
        #  return self.sigmoid(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x1, x2, y = train_batch
        y_hat = self.forward(x1, x2)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x1, x2, y = val_batch
        y_hat = self.forward(x1, x2)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

    @pl.data_loader
    def train_dataloader(self):
        ## todo: complete dataloader
        pass
        # return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams["batch_size"])

    @pl.data_loader
    def val_dataloader(self):
        ## todo: complete dataloader
        pass
        # return DataLoader(self.val_dataset, batch_size=self.hparams["batch_size"])

