import torch
import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):

    def __init__(self, mode):
        super().__init__()
        self.out_ch = 2048

        resnet = models.resnet50(pretrained=True)
        if mode == 'feature-extract':
            for child in resnet.children():
                for param in child.parameters():
                    param.requires_grad = False

        # remove fc, avgpool layer from resnet
        self.encoder = nn.Sequential()
        for name, module in resnet.named_children():
            if name == 'fc' or name == 'avgpool':
                continue
            self.encoder.add_module(name, module)

    def forward(self, x):
        return self.encoder(x)


class ChestXrayClassifier(nn.Module):

    def __init__(self, img_size=256, n_class=15, mode='feature-extract',
                 backbone='resnet50', pooling_type='max'):
        super().__init__()
        self.backbone = None
        self.pool = None
        self.S = {256: 8, 512: 16, 1024: 32}[img_size]

        if backbone == 'resnet50':
            self.backbone = ResNet50(mode)
        else:
            raise ValueError(f'Unrecognized backbone: {backbone}')

        if pooling_type == 'max':
            self.pool = nn.MaxPool2d(kernel_size=2)
        else:
            raise ValueError(f'Unrecognized pooling_type: {pooling_type}')

        self.classifier = nn.Linear(int(self.S/2 * self.S/2 * self.backbone.out_ch), n_class)
        torch.nn.init.xavier_uniform_(self.classifier.weight.data)
        torch.nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        batch_size, _, H, W = x.shape

        # expand images to 3 channels since resnet expects 3 channels
        x = x.expand(batch_size, 3, H, W)

        # Pretrained network
        x = self.backbone(x)
        x = x.view(batch_size, self.backbone.out_ch, self.S, self.S)

        # Pooling layer
        x = self.pool(x)

        # Fully-connected output layer
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x


def test():
    print('Testing ChestXrayClassifier')
    device = 'cpu'
    mode = 'feature-extract'
    backbone = 'resnet50'
    n_class = 15
    img_size = 256
    model = ChestXrayClassifier(img_size, n_class, mode, backbone).to(device)
    batch_size = 4
    num_ch = 1
    x = torch.randn(batch_size, num_ch, img_size, img_size).to(device)
    out = model(x)
    assert out.shape == (batch_size, n_class)
    print('Test passed!')


if __name__ == '__main__':
    test()
