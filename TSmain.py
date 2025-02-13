#TSmain.py
import torch
import os
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from TSmodel import T_teacher_net, R_teacher_net, student_net
from data_loader import get_loader_t, get_loader_r
import pytorch_ssim
import lpips
import time
import torchvision.models as models
import torch.nn.functional as F
from torchvision.utils import save_image
import pandas as pd
from skimage import color as skimage_color
from torch.nn import L1Loss

class MultipleLoss(torch.nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = torch.nn.ModuleList(losses)
        self.weight = weight or [1 / len(self.losses)] * len(self.losses)

    def forward(self, predict, target):
        Mloss = 0
        for weight, loss in zip(self.weight, self.losses):
            Mloss += loss(predict, target) * weight
        return Mloss

class HSVLoss(torch.nn.Module):
    def __init__(self):
        super(HSVLoss, self).__init__()

    def forward(self, predict, target):

        predict_hsv = skimage_color.rgb2hsv(predict.detach().cpu().numpy().transpose(0, 2, 3, 1))
        target_hsv = skimage_color.rgb2hsv(target.detach().cpu().numpy().transpose(0, 2, 3, 1))
        predict_hsv = torch.tensor(predict_hsv, device=predict.device).permute(0, 3, 1, 2)[:, 0:1, :, :]
        target_hsv = torch.tensor(target_hsv, device=target.device).permute(0, 3, 1, 2)[:, 0:1, :, :]

        diff = torch.abs(predict_hsv - target_hsv)
        mask = (diff > 0.5).float()
        mask_ = (0.5 > diff).float()

        hsvloss = torch.mean(torch.abs(diff * mask_)) + torch.mean(torch.abs(diff * mask - 0.5 * mask))
        return hsvloss

class SCDLoss(torch.nn.Module): #StandardColorDifferenceLoss 標準色差損失
    def __init__(self):
        super(SCDLoss, self).__init__()
        #self.l2_loss = torch.nn.MSELoss()
        self.hsv_loss = HSVLoss()
        self.M_loss = MultipleLoss([L1Loss(), self.hsv_loss])

    def forward(self, T, R, GT_T, GT_R):
        # 如果影像在 [0, 255] RGB 範圍內，則將影像標準化為 [0, 1]
        # 以groundtruth為標準參考計算L1損失
        #l2_loss_T = self.l2_loss(T, GT_T) # 預測T與標準GT(T)之間的差異
        #l2_loss_R = self.l2_loss(R, GT_R) # 預測 R 和標準 GT(R) 之間的差異
        #combined_loss = (l2_loss_T + l2_loss_R) / 2.0 # 組合損失為 l1_loss_T 和 l1_loss_R 的平均值
        T = T / 255.0
        R = R / 255.0
        GT_T = GT_T / 255.0
        GT_R = GT_R / 255.0
        l1_loss_T = F.l1_loss(T, GT_T)
        l1_loss_R = F.l1_loss(R, GT_R)
        hsv_loss_T = self.hsv_loss(T, GT_T)
        hsv_loss_R = self.hsv_loss(R, GT_R)
        M_loss_T = self.M_loss(T, GT_T)
        M_loss_R = self.M_loss(R, GT_R)
        #psnr_T = self.psnrSCD(T, GT_T)
        #psnr_R = self.psnrSCD(R, GT_R)

        scd_loss = (l1_loss_T + l1_loss_R + hsv_loss_T + hsv_loss_R + M_loss_T + M_loss_R) / 6.0 #- (psnr_T + psnr_R) / 2.0

        return scd_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssim_loss = pytorch_ssim.SSIM()
SI_loss_fn = pytorch_ssim.SSIM_SI_SIP(size_average=False).to(device)
loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
color_loss = SCDLoss()

vgg16 = models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1").features.to(device)
for param in vgg16.parameters():
    param.requires_grad = False

def compute_psnr(input, target):
    mse = F.mse_loss(input, target)
    return 10 * torch.log10(1.0 / mse)

def compute_perceptual_loss(vgg, input, target):
    features_input = vgg(input)
    features_target = vgg(target)
    return F.l1_loss(features_input, features_target)


def train_T_teacher(config):
    model = T_teacher_net().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.learn_T1, betas=(config.b1, config.b2))
    writer = SummaryWriter(config.log_dir_T)
    loader_t = get_loader_t(config)

    epoch_data = []

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()

        for i, (inputs, GT, alpha) in enumerate(loader_t):
            inputs, GT = inputs.to(device), GT.to(device)
            c = torch.tensor([0.5]).to(device).expand(inputs.size(0))

            try:
                _, _, out0, out1, out2, outputs = model(inputs, c)
                Pre_T = outputs[:, :3, :, :]

                # Compute losses
                ssim_value = ssim_loss(Pre_T, GT)
                psnr_value = compute_psnr(Pre_T, GT)
                perceptual_loss_value = compute_perceptual_loss(vgg16, Pre_T, GT)
                lpips_loss_value = loss_fn_lpips(Pre_T, GT).mean()

                # Compute SI loss
                _, si_value, _ = SI_loss_fn(Pre_T, GT)
                si_value = si_value.mean() if si_value.ndim > 0 else si_value

                # Total loss with SI loss included
                total_loss = -ssim_value - si_value - (psnr_value / 40) + 3 + 2 * perceptual_loss_value + lpips_loss_value

                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Logging
                writer.add_scalar('Loss/Total', total_loss.item(), epoch)
                writer.add_scalar('Loss/SSIM', ssim_value.item(), epoch)
                writer.add_scalar('Loss/SI', si_value.item(), epoch)
                writer.add_scalar('Loss/PSNR', psnr_value.item(), epoch)
                writer.add_scalar('Loss/Perceptual', perceptual_loss_value.item(), epoch)
                writer.add_scalar('Loss/LPIPS', lpips_loss_value.item(), epoch)

                if i % config.log_step == 0:
                    print(f"Epoch [{epoch}/{config.num_epochs}], Step [{i}/{len(loader_t)}], "
                          f"SSIM: {ssim_value:.6f}, SI: {si_value:.6f}, PSNR: {psnr_value:.6f}, "
                          f"Perceptual: {perceptual_loss_value:.6f}, LPIPS: {lpips_loss_value:.6f}")
            except Exception as e:
                print(f"Error during forward pass: {e}")
                continue

        save_image(GT, os.path.join(config.sample_dir, f'1epoch_{epoch}_T_GT.png'))
        save_image(inputs, os.path.join(config.sample_dir, f'1epoch_{epoch}_T_Input.png'))
        save_image(Pre_T, os.path.join(config.sample_dir, f'1epoch_{epoch}_T_Pre.png'))

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch}/{config.num_epochs}] completed in {epoch_duration:.2f} seconds")

        epoch_data.append({
            'Epoch': epoch,
            'Total Loss': total_loss.item(),
            'SSIM': ssim_value.item(),
            'SI': si_value.item(),
            'PSNR': psnr_value.item(),
            'Perceptual Loss': perceptual_loss_value.item(),
            'LPIPS Loss': lpips_loss_value.item(),
            'Runtime (seconds)': epoch_duration
        })

        if epoch % config.save_step == 0:
            torch.save(model.state_dict(), os.path.join(config.model_save_dir, f'T_teacher_epoch_{epoch}.pth'))
    writer.close()

def train_R_teacher(config):
    print("Training R_teacher network...")
    model = R_teacher_net().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.learn_R1, betas=(config.b1, config.b2))
    writer = SummaryWriter(config.log_dir_R)
    loader_r = get_loader_r(config)

    epoch_data = []

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()

        for i, (inputs, GT, alpha) in enumerate(loader_r):
            inputs, GT = inputs.to(device), GT.to(device)
            c = torch.tensor([0.5]).to(device).expand(inputs.size(0))

            try:

                out0, out1, out2, outputs = model(inputs, c)
                Pre_R = outputs[:, :3, :, :]  # predict

                ssim_value, si_value, _ = SI_loss_fn(Pre_R, GT)
                si_value = si_value.mean() if si_value.ndim > 0 else si_value

                ssim_value = ssim_loss(Pre_R, GT)
                si_value = torch.mean((GT - Pre_R) ** 2)
                psnr = compute_psnr(Pre_R, GT)
                perceptual = compute_perceptual_loss(vgg16, Pre_R, GT)
                lpips = loss_fn_lpips(Pre_R, GT).mean()

                #color_loss = F.l1_loss(Pre_R.mean(dim=(2, 3)), GT.mean(dim=(2, 3)))
                #color_loss = color_loss(Pre_R, Pre_R, GT, GT)
                #Mloss = MultipleLoss(Pre_R, GT)

                total_loss = -ssim_value - si_value - (psnr / 40) + 3 + 2 * perceptual + lpips #+ Mloss   #  color_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                writer.add_scalar('Loss/Total', total_loss.item(), epoch)
                writer.add_scalar('Loss/SSIM', ssim_value.item(), epoch)
                writer.add_scalar('Loss/SI', si_value.item(), epoch)
                writer.add_scalar('Loss/PSNR', psnr.item(), epoch)
                writer.add_scalar('Loss/Perceptual', perceptual.item(), epoch)
                writer.add_scalar('Loss/LPIPS', lpips.item(), epoch)
                #writer.add_scalar('Loss/MLoss', MLoss.item(), epoch)
                #writer.add_scalar('Loss/Color', color_loss.item(), epoch)

                if i % config.log_step == 0:
                    print(
                        f"Epoch [{epoch}/{config.num_epochs}], Step [{i}/{len(loader_r)}], "
                        f"SSIM: {ssim_value:.6f}, SI: {si_value:.6f}, PSNR: {psnr:.6f}, "
                        f"Perceptual: {perceptual:.6f}, LPIPS: {lpips:.6f}, "
                    )
            except Exception as e:
                print(f"Error during forward pass: {e}")
                continue

        save_image(GT, os.path.join(config.sample_dir, f'2epoch_{epoch}_R_GT.png'))
        save_image(inputs, os.path.join(config.sample_dir, f'2epoch_{epoch}_R_Input.png'))
        save_image(Pre_R, os.path.join(config.sample_dir, f'2epoch_{epoch}_R_Pre.png'))

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch}/{config.num_epochs}] completed in {epoch_duration:.2f} seconds")

        epoch_data.append({
            'Epoch': epoch,
            'Total Loss': total_loss.item(),
            'SSIM': ssim_value.item(),
            'SI': si_value.item(),
            'PSNR': psnr.item(),
            'Perceptual Loss': perceptual.item(),
            'LPIPS Loss': lpips.item(),

            'Runtime (seconds)': epoch_duration
        })

        if epoch % config.save_step == 0:
            torch.save(model.state_dict(), os.path.join(config.model_save_dir, f'R_teacher_epoch_{epoch}.pth'))

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_dir', type=str, default='E:/CHAN/absorption_2/main/matlab/estimate_ablation_study/training_data/')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learn_T1', type=float, default=0.0001, help='Learning rate for T_teacher')
    parser.add_argument('--learn_R1', type=float, default=0.0001, help='Learning rate for R_teacher')
    #parser.add_argument('--learn_S', type=float, default=0.0001, help='Learning rate for student')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--b1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--b2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    parser.add_argument('--log_dir_T', type=str, default='E:/CHAN/absorption_2/main/GFLO70148_ablationstudy/tea_2_normal/logs_T')
    parser.add_argument('--log_dir_R', type=str, default='E:/CHAN/absorption_2/main/GFLO70148_ablationstudy/tea_2_normal/logs_R')
    #parser.add_argument('--log_dir_S', type=str, default='E:/CHAN/absorption_2/main/TS/logs_S')
    parser.add_argument('--model_save_dir', type=str, default='E:/CHAN/absorption_2/main/GFLO70148_ablationstudy/tea_2_normal/Tmodels')
    parser.add_argument('--sample_dir', type=str, default='E:/CHAN/absorption_2/main/GFLO70148_ablationstudy/tea_2_normal/teasample')
    parser.add_argument('--log_step', type=int, default=10, help='Logging frequency')
    parser.add_argument('--save_step', type=int, default=10, help='Model saving frequency')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU workers for data loading')
    parser.add_argument('--student_checkpoint', type=str, default=None, help='Path to student network checkpoint')

    config = parser.parse_args()

    print("Training T_teacher network...")
    train_T_teacher(config)
    print("Training R_teacher network...")
    train_R_teacher(config)

