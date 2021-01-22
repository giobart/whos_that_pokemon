from src.model.GeneralLayers import *
from facenet_pytorch import InceptionResnetV1
from src.model.inception_bn import bn_inception

class myCNN(nn.Module):
    """Small CNN used to test the setup"""
    def __init__(self, activ_fn, f_size, f_channels, padding=0):
        super().__init__()
        input_channels, height, width = (3, 128, 128)
        self.input_size = height
        self.output_size = int(f_channels * 4 * int(height / 8) * int(width / 8))

        self.model = nn.Sequential(
            ConvPoolPRelu(input_channels, f_channels, f_size, padding, activ_fn=activ_fn),
            ConvPoolPRelu(f_channels, f_channels * 2, f_size, padding, activ_fn=activ_fn),
            ConvPoolPRelu(f_channels * 2, f_channels * 4, f_size, padding, activ_fn=activ_fn),
        )

    def forward(self, x):
        return self.model(x)

class InceptionRNV1(nn.Module):
    """Inception Resnet V1 network used to train the model implementing the Contrastive Loss."""
    def __init__(self, freeze_layers):
        super().__init__()
        self.input_size = 224
        self.output_size = 1792

        cnn = InceptionResnetV1(pretrained='vggface2', classify=False)

        if freeze_layers:
            cnn.requires_grad_(False)

        self.model = nn.Sequential(*list(cnn.children())[:-4])
        self.model.eval()

    def forward(self, x):
        return self.model(x)

class BnInception(nn.Module):
    """BN inception model used to train the model implementing the Group Loss"""
    def __init__(self, num_classes=1000, pretrained=True):
        """

        :param num_classes: num_classes will influence the last classification layer.
        :param pretrained:  whether to load the pretrained ImageNet weights or not
        """
        super().__init__()
        self.input_size = 224
        self.output_size = num_classes

        self.model = bn_inception(pretrained=pretrained)
        self.model.last_linear = nn.Linear(1024, num_classes)
        # if not finetune and (weights_path is not None or cnn_state_dict is not None):
        #     self.model.load_state_dict(cnn_state_dict)

    def forward(self, x):
        return self.model(x)