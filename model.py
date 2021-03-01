import torch
import  torch.nn as nn
import torchvision.transforms.functional as TF

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
        )

    def forward(self, x):

        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, **kwargs):
        super(UNET, self).__init__()


        features = [64, 128, 256, 512]


        self.down_sampling = nn.ModuleList()
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.up_sampling = nn.ModuleList()


        for feature in features:
            self.down_sampling.append(CNN(in_channels, feature))
            in_channels = feature

        for feature in  reversed(features):
            self.up_sampling.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, ))
            self.up_sampling.append(CNN(feature*2, feature))

        self.bottleneck = CNN(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels , kernel_size=1)

    def forward(self, x):
        skip_connections = []

        #down-sampling
        for down in self.down_sampling:
            x = down(x)
            skip_connections.append(x)
            x = self.max_pooling(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        #up-sampling
        for up in range(0, len(self.up_sampling), 2):
            x = self.up_sampling[up](x)
            skip_connection = skip_connections[up//2]

            if x.shape != skip_connection.shape :
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up_sampling[up + 1](concat_skip)

        return self.final_conv(x)

