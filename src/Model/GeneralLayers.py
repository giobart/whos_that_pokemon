from torch import nn


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