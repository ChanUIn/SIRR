#student.py
import torch
import os
import time
import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from TSmodel import T_teacher_net, R_teacher_net, student_net
from data_loader import get_loader_t, get_loader_r
from torchvision.utils import save_image
import pytorch_ssim
import lpips
import torchvision.models as models
import torch.nn.functional as F
from torch.nn import L1Loss
from skimage import color as skimage_color

def psnr(input, target):
    psnr = ((input - target) ** 2).mean()
    psnr = 10 * (torch.log10(1.0 / psnr))
    return psnr

def compute_perceptual_loss(vgg, input, target):
    features_input = vgg(input)
    features_target = vgg(target)
    perceptual_loss = F.l1_loss(features_input, features_target)
    return perceptual_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssim_loss = pytorch_ssim.SSIM(window_size=11)
loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
SI_loss_fn = pytorch_ssim.SSIM_SI_SIP(size_average=False).to(device)
L1_loss = torch.nn.L1Loss()


vgg16 = models.vgg16(pretrained=True).features.to(device)
for param in vgg16.parameters():
    param.requires_grad = False

def feature_matching_loss(student_features, teacher_features):
    return sum(L1_loss(s, t) for s, t in zip(student_features, teacher_features))

def attention_map(features):
    return torch.sum(features, dim=1, keepdim=True)

def attention_loss(student_features, teacher_features):
    student_map = attention_map(student_features)
    teacher_map = attention_map(teacher_features)
    return L1_loss(student_map, teacher_map)


def train_T_student(config):
    print("Training T_student network...")

    model = student_net().to(device)
    model_T = T_teacher_net().to(device)

    model_T.load_state_dict(torch.load(os.path.join(config.model_save_dir, 'T_teacher_epoch_200.pth')), strict=False)
    model_T.eval()
    for param in model_T.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learn_T1, betas=(config.b1, config.b2))
    writer = SummaryWriter(config.log_dir_T_student)
    loader_t = get_loader_t(config)

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()
        model.train()

        for i, (inputs, GT, alpha) in enumerate(loader_t):
            inputs, GT = inputs.to(device), GT.to(device)
            c = torch.tensor([0.5]).to(device).expand(inputs.size(0))
            with torch.no_grad():
                _, _, teacher_features, _, _, outputs_T_teacher = model_T(inputs, c)

            _, _, _, _, _, student_features, student_intermediate, final_out_T = model(inputs, c)
            pre_T = final_out_T[:, :3, :, :]


            ssim_TS = ssim_loss(pre_T, GT)
            psnr_TS = psnr(pre_T, GT)
            perceptual_TS = compute_perceptual_loss(vgg16, pre_T, GT)
            lpips_TS = loss_fn_lpips(pre_T, GT).mean()
            #feature_match_loss = feature_matching_loss(student_features, teacher_features)
            #attention_map_loss = attention_loss(student_intermediate, teacher_features)

            _, si_value, _ = SI_loss_fn(pre_T, GT)
            si_value = si_value.mean() if si_value.ndim > 0 else si_value

            total_loss = -ssim_TS - si_value -(psnr_TS / 40)+ 3+ 2 * perceptual_TS + lpips_TS #+ 0.1 * feature_match_loss  + 0.1 * attention_map_loss  )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            global_step = (epoch - 1) * len(loader_t) + i
            writer.add_scalar('Train_T_student/Loss_Total', total_loss.item(), global_step)
            writer.add_scalar('Train_T_student/Loss_TSSSIM', ssim_TS.item(), global_step)
            writer.add_scalar('Train_T_student/Loss_TSSI', si_value.item(), global_step)
            writer.add_scalar('Train_T_student/Loss_TSPSNR', psnr_TS.item(), global_step)
            writer.add_scalar('Train_T_student/Loss_TSPerceptual', perceptual_TS.item(), global_step)
            writer.add_scalar('Train_T_student/Loss_TSLPIPS', lpips_TS.item(), global_step)
            #writer.add_scalar('Train_T_student/Loss_FeatureMatch', feature_match_loss.item(), global_step)
            #writer.add_scalar('Train_T_student/Loss_AttentionMap', attention_map_loss.item(), global_step)

            if i % config.log_step == 0:
                print(f"Epoch [{epoch}/{config.num_epochs}], Step [{i}/{len(loader_t)}], "
                      f"TSSSIM: {ssim_TS:.4f}, TSSI: {si_value:.4f}, TSPSNR: {psnr_TS:.4f}, TSPerceptual: {perceptual_TS:.4f}, "
                      f"TSLPIPS: {lpips_TS:.4f}")#, FeatureMatch: {feature_match_loss:.4f}, AttentionMap: {attention_map_loss:.4f}")

        save_image(pre_T, os.path.join(config.sample_dir, f'1epoch_{epoch}_pre_T_student.png'))
        save_image(inputs, os.path.join(config.sample_dir, f'1epoch_{epoch}_input_T_student.png'))
        save_image(GT, os.path.join(config.sample_dir, f'1epoch_{epoch}_GT_T_student.png'))

        if epoch % config.save_step == 0:
            torch.save(model.state_dict(), os.path.join(config.model_save_dir, f'T_student_epoch_{epoch}.pth'))

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch}/{config.num_epochs}] completed in {epoch_duration:.2f} seconds")

    writer.close()
    print("Finished Training T_student network")


def train_R_student(config):
    print("Training R_student network...")

    model = student_net().to(device)
    model_R = R_teacher_net().to(device)
    model_R.load_state_dict(torch.load(os.path.join(config.model_save_dir, 'R_teacher_epoch_200.pth')), strict=False)
    model_R.eval()
    for param in model_R.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learn_R1, betas=(config.b1, config.b2))
    writer = SummaryWriter(config.log_dir_R_student)
    loader_r = get_loader_r(config)
    #multiple_loss = MultipleLoss([L1_loss, torch.nn.MSELoss()]).to(device)

    for epoch in range(1, config.num_epochs + 1):
        epoch_start_time = time.time()
        model.train()

        for i, (inputs, GT, alpha) in enumerate(loader_r):
            inputs, GT = inputs.to(device), GT.to(device)
            c = torch.tensor([0.5]).to(device).expand(inputs.size(0))

            with torch.no_grad():
                _, _, _, outputs_R_teacher = model_R(inputs, c) # teacher net output

            _, _, _, _, _, _, student_features, final_out_R = model(inputs, c) # Student outputs
            pre_R = final_out_R[:, :3, :, :]

            sim_value, si_value, _ = SI_loss_fn(pre_R, GT)

            ssim_RS = ssim_loss(pre_R, GT)
            si_value = torch.mean((GT - pre_R) ** 2)
            psnr_RS = psnr(pre_R, GT)
            perceptual_RS = compute_perceptual_loss(vgg16, pre_R, GT)
            lpips_RS = loss_fn_lpips(pre_R, GT).mean()

            total_loss = -ssim_RS - si_value - (psnr_RS / 40) + 3 + 2 * perceptual_RS + lpips_RS #+ Mul_RS + Res_RS #+ color_loss_R
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            global_step = (epoch - 1) * len(loader_r) + i
            writer.add_scalar('Train_R_student/Loss_Total', total_loss.item(), global_step)
            writer.add_scalar('Train_R_student/Loss_RSSSIM', ssim_RS.item(), global_step)
            writer.add_scalar('Train_R_student/Loss_RSPSNR', psnr_RS.item(), global_step)
            writer.add_scalar('Train_R_student/Loss_RSPerceptual', perceptual_RS.item(), global_step)
            writer.add_scalar('Train_R_student/Loss_RSLPIPS', lpips_RS.item(), global_step)
            writer.add_scalar('Train_R_student/Loss_SI', si_value.item(), global_step)
            #writer.add_scalar('Train_R_student/Loss_Mul', Mul_RS.item(), global_step)
            #writer.add_scalar('Train_R_student/Loss_Res', Res_RS.item(), global_step)
            #writer.add_scalar('Train_R_student/Loss_StandardColorDifference', color_loss_R.item(), global_step)

            if i % config.log_step == 0:
                print(f"Epoch [{epoch}/{config.num_epochs}], Step [{i}/{len(loader_r)}], "
                      f"RSSSIM: {ssim_RS:.4f}, RSPSNR: {psnr_RS:.4f}, RSPerceptual: {perceptual_RS:.4f}, "
                      f"RSLPIPS: {lpips_RS:.4f}, SI: {si_value:.4f}")#, SI: {si_value:.4f}")#, Mul: {Mul_RS:.4f}, Res: {Res_RS:.4f}")#, StandardColorDiff: {color_loss_R:.4f}" )

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch [{epoch}/{config.num_epochs}] completed in {epoch_duration:.2f} seconds")

        save_image(pre_R, os.path.join(config.sample_dir, f'2epoch_{epoch}_pre_student.png'))
        save_image(inputs, os.path.join(config.sample_dir, f'2epoch_{epoch}_input_student.png'))
        save_image(GT, os.path.join(config.sample_dir, f'2epoch_{epoch}_GT_student.png'))

        if epoch % config.save_step == 0:
            torch.save(model.state_dict(), os.path.join(config.model_save_dir, f'R_student_epoch_{epoch}.pth'))

    writer.close()
    print("Finished Training R_student network")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_dir', type=str, default='E:/CHAN/absorption_2/main/matlab/estimate_ablation_study/training_data/')# 5000traininigdataset
    #parser.add_argument('--main_dir', type=str, default='D:/CHAN/absorption/main/matlab/dataset/training/training_data_NF/')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learn_T1', type=float, default=0.0001, help='Learning rate for T_student')
    parser.add_argument('--learn_R1', type=float, default=0.0001, help='Learning rate for R_student')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--b1', type=float, default=0.5, help='Beta1 for Adam optimizer')
    parser.add_argument('--b2', type=float, default=0.999, help='Beta2 for Adam optimizer')
    parser.add_argument('--log_dir_T_student', type=str, default='E:/CHAN/absorption_2/main/GFLO70148_ablationstudy/tea_normal_stu_onlyFEF_shallow/logs_Tstu')
    parser.add_argument('--log_dir_R_student', type=str, default='E:/CHAN/absorption_2/main/GFLO70148_ablationstudy/tea_normal_stu_onlyFEF_shallow/logs_Rstu')
    parser.add_argument('--model_save_dir', type=str, default='E:/CHAN/absorption_2/main/GFLO70148_ablationstudy/tea_normal_stu_onlyFEF_shallow/Tmodels')
    parser.add_argument('--sample_dir', type=str, default='E:/CHAN/absorption_2/main/GFLO70148_ablationstudy/tea_normal_stu_onlyFEF_shallow/stu_sam')
    parser.add_argument('--log_step', type=int, default=10, help='Logging frequency')
    parser.add_argument('--save_step', type=int, default=10, help='Model saving frequency')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU workers for data loading')

    config = parser.parse_args()

    print("Training T_student network...")
    train_T_student(config)
    print("Training R_student network...")
    train_R_student(config)
#

