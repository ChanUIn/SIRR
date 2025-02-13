# modulegroup.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FE2(nn.Module):
    def __init__(self, in_channels=3, feature_dim=32, num_blocks=6):
        super(FE2, self).__init__()
        layers = [nn.Conv2d(in_channels, feature_dim, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_blocks):
            layers.append(RB(feature_dim, feature_dim))
        self.fea_ex = nn.Sequential(*layers)

    def forward(self, x):
        return self.fea_ex(x)

class FEF(nn.Module):
    """Feature extraction and fusion module from shallow features to structural features"""
    def __init__(self, in_channels=3, conv_dim=64, repeat_num=12):
        super(FEF, self).__init__()
        layers = [
            nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(inplace=True),
        ]
        curr_dim = conv_dim
        #for _ in range(repeat_num):
        #    layers.append(RCAB(in_channels=curr_dim))
        for _ in range(repeat_num):
            layers.append(RB(dim_in=curr_dim, dim_out=curr_dim))
        layers.append(nn.Conv2d(conv_dim, curr_dim, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(curr_dim))
        layers.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        return out

class BA(nn.Module): # Brightness Attention
    def __init__(self, in_channels):
        super(BA, self).__init__()
        #self.avg_pool = nn.AvgPool2d(1)
        self.max_pool = nn.MaxPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #brightness = self.avg_pool(x)   # brightness = self.max_pool(x) (GFLO 701.48 ) change to avgpooling (GFLO 701.51 )
        brightness = self.max_pool(x)
        brightness = brightness.expand_as(x)
        attention_map = self.sigmoid(self.conv(brightness))
        return x * attention_map

class RB(nn.Module):
    """Residual Block"""
    def __init__(self, dim_in, dim_out):
        super(RB, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class RSM(nn.Module):
    """Reflection Separation Module"""
    def __init__(self, conv_dim=64, repeat_num=12):
        super(RSM, self).__init__()

        layers = [
            nn.Conv2d(65, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(inplace=True)
        ]
        curr_dim = conv_dim

        for _ in range(repeat_num):
            layers.append(RCAB(in_channels=curr_dim))

        for _ in range(repeat_num):
            layers.append(RB(dim_in=curr_dim, dim_out=curr_dim))

        layers.append(nn.Conv2d(curr_dim, 64, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.ReLU6())

        self.main = nn.Sequential(*layers)

    def forward(self, x, c):

        if c.dim() == 1:
            c = c.view(c.size(0), 1, 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        return self.main(x)

class RCAB(nn.Module):
    """Residual Channel Attention Block"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(RCAB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)  # 移除 conv3 -39.01
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)  # 移除 conv3 -39.01
        out = self.relu(out)
        max_pool = self.max_pool(out).view(out.size(0), -1)
        max_pool = self.fc2(self.relu(self.fc1(max_pool)))
        max_pool = self.sigmoid(max_pool).unsqueeze(-1).unsqueeze(-1)
        out = out * max_pool
        out += residual

        return out

class Downsampling(nn.Module):

    def __init__(self):
        super(Downsampling, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.main(x)

class Downsampling_T(nn.Module):

    def __init__(self):
        super(Downsampling_T, self).__init__()
        self.Layer1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.Layer2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.Layer1(x)
        out = self.relu(out)
        out = self.Layer2(out)
        return self.relu(out)

class Upsampling_T(nn.Module):

    def __init__(self):
        super(Upsampling_T, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

class backbone(nn.Module):

    def __init__(self):
        super(backbone, self).__init__()
        self.Layer1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.Layer2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):

        out1 = self.Layer1(x)
        out2 = self.relu(out1)
        return self.Layer2(out2) + x

def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '12': 'conv4_1',
                  '19': 'conv5_1',
                  '23': 'conv6_1',
                  '28': 'conv7_1',
                  '34': 'conv8_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def get_model():
    vgg19 = models.vgg19(pretrained=True).features
    for param in vgg19.parameters():
        param.requires_grad_(False)
    return vgg19

# RB, RCAB, FE, RSM, LM, RIR, Downsampling, Downsampling_T, Upsampling_T, backbone, SPBlock, get_features, get_model
#