#TSmodel.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modulegroup import RB, FEF, RSM, Downsampling, Downsampling_T, Upsampling_T, backbone, FE2, BA, RCAB

class Channel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Channel, self).__init__()
        self.adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.adjust(x)

class T_teacher_net(nn.Module):
    def __init__(self):
        super(T_teacher_net, self).__init__()
        self.down = Downsampling_T()
        self.ResBlock1 = backbone()
        self.ResBlock2 = backbone()
        self.ResBlock3 = backbone()
        self.ResBlock4 = backbone()
        self.RB = nn.ModuleList([RB(64, 64) for _ in range(2)])
        self.fe = FEF(in_channels=64, conv_dim=64, repeat_num=6)
        self.rsm = RSM(conv_dim=64, repeat_num=3)
        self.up = Upsampling_T()

    def forward(self, x, c):
        out0 = self.down(x)
        out1 = self.ResBlock1(out0)
        out2 = self.ResBlock2(out1)
        out3 = self.ResBlock3(out2)
        out4 = self.ResBlock4(out3)
        for rb in self.RB:
            out1 = rb(out4)

        features = self.fe(out4)
        rsm_Tout = self.rsm(features, c)
        final_Tout = self.up(rsm_Tout)

        return out1, out2, out3, out4, self.up(rsm_Tout), final_Tout


class R_teacher_net(nn.Module):
    def __init__(self):
        super(R_teacher_net, self).__init__()
        self.down = Downsampling_T()
        self.fea_ex = FE2(in_channels=3, feature_dim=32, num_blocks=1)
        self.channel = Channel(in_channels=32, out_channels=64)
        self.backbone = nn.ModuleList([backbone() for _ in range(4)])  # ori 4
        self.brightness_attention = nn.ModuleList([BA(in_channels=64) for _ in range(3)])
        self.RB = nn.ModuleList([RB(64, 64) for _ in range(2)])# ori 4
        self.fe = FEF(in_channels=64, conv_dim=64, repeat_num=2)# ori 6
        self.rsm = RSM(conv_dim=64, repeat_num=2)# ori 6
        self.up = Upsampling_T()

    def forward(self, x, c):
        out0 = self.down(x)
        features = self.fea_ex(x)
        features = self.channel(features)
        out1 = features + out0

        for back, ba in zip(self.backbone, self.brightness_attention):
            out1 = back(out1)
            out1 = ba(out1)

        for rb in self.RB:
            out1 = rb(out1)

        features = self.fe(out1)
        rsm_Rout = self.rsm(features, c)
        final_Rout = self.up(rsm_Rout)

        return out0, out1, self.up(rsm_Rout), final_Rout

class student_net(nn.Module):
    def __init__(self):
        super(student_net, self).__init__()
        self.down = Downsampling()
        self.channel_reduce = nn.Conv2d(128, 3, kernel_size=1, bias=False)
        self.fea_ex = FEF(in_channels=3, conv_dim=32, repeat_num=2)  # 共享特徵提取器  FEF_shallow = FE2 change tp FEF_Fusion
        self.channel = Channel(in_channels=32, out_channels=64)
        self.T_block1 = RB(64, 64)  # T 特定層
        self.T_block2 = RB(64, 64)
        self.T_block3 = RB(64, 64)
        self.T_block4 = RB(64, 64)
        self.R_block1 = RB(64, 64)  # R 特定層
        self.R_block2 = RB(64, 64)
        self.R_block3 = RB(64, 64)
        self.R_block4 = RB(64, 64)
        self.shared_RB = nn.ModuleList([RB(64, 64) for _ in range(6)])  # 共享 residual blocks
        #self.dropout = nn.Dropout(p=0.2) ##### dropout
        self.fe = FEF(in_channels=64, conv_dim=64, repeat_num=6)  # 附加特徵處理
        self.rcab = RCAB(in_channels=64)
        self.rsm = RSM(conv_dim=64, repeat_num=2)
        self.up_T = Upsampling_T()
        self.up_R = Upsampling_T()
        #self.dropout = nn.Dropout(p=0.2) ## dropout

    def forward(self, x, c):
        out0 = self.down(x)
        reduced_out0 = self.channel_reduce(out0)
        features = self.fea_ex(reduced_out0)  # 提取特徵並調整channels
        features = self.channel(features)
        shared_out = features

        for rb in self.shared_RB:
            shared_out = rb(shared_out)

        #shared_out = self.dropout(shared_out) ### dropout
        en_features = self.rcab(shared_out)
        rsm = self.rsm(en_features, c)
        #rsm = self.dropout(rsm)  ## dropout

        out_T1 = self.T_block1(rsm)
        out_T2 = self.R_block2(out_T1)
        out_T3 = self.R_block2(out_T2)
        final_out_T = self.up_T(out_T3)

        out_R1 = self.R_block1(en_features)
        out_R2 = self.R_block2(out_R1)
        out_R3 = self.R_block2(out_R2)
        final_out_R = self.up_R(out_R3)


        return out0, shared_out, out_T1, final_out_T, out_R1, out_R2, out_R3, final_out_R

def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '2': '1',
            '7': '2',
            '12': '3',
            '21': '4',
            '30': '5',
        }

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
#
