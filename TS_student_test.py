import os
import torch
import pandas as pd
import pytorch_ssim
import lpips
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from TSmodel import student_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ssim = pytorch_ssim.SSIM()
lpips = lpips.LPIPS(net='vgg').to(device)

t_model_path = "E:/CHAN/absorption_2/main/GFLO70148_ablationstudy/normal_only_T_teacher/Tmodels/T_student_epoch_200.pth"
r_model_path = "E:/CHAN/absorption_2/main/GFLO70148_ablationstudy/normal_only_T_teacher/Tmodels/R_student_epoch_200.pth"
#test_data_dir = "E:/CHAN/absorption_2/main/matlab/dataset/test_NC/18/syn/"
test_data_dir = "E:/CHAN/absorption_2/main/matlab/dataset/test/18/syn"
output_excel_path = "E:/CHAN/absorption_2/main/GFLO70148_ablationstudy/normal_only_T_teacher/only_R_teacher.xlsx"
output_test_dir = "E:/CHAN/absorption_2/main/GFLO70148_ablationstudy/normal_only_T_teacher/only_R_teacher"
os.makedirs(output_test_dir, exist_ok=True)

t_model = student_net().to(device)
r_model = student_net().to(device)
t_model.load_state_dict(torch.load(t_model_path))
r_model.load_state_dict(torch.load(r_model_path))

t_model.eval()
r_model.eval()

vgg16 = models.vgg16(pretrained=True).features.to(device)
for param in vgg16.parameters():
    param.requires_grad = False

transform = transforms.Compose([
    #transforms.Resize((256, 256)),
    transforms.ToTensor()
])


test_images = [os.path.join(test_data_dir, img) for img in os.listdir(test_data_dir) if img.endswith(('.png', '.jpg'))]


def lmse(input, target, window_size=7):
    """
    Compute Local Mean Square Error (LMSE).
    :param input: Predicted image tensor [1, C, H, W]
    :param target: Ground Truth image tensor [1, C, H, W]
    :param window_size: Size of the local window
    """
    pad = window_size // 2
    input_mean = F.avg_pool2d(input, kernel_size=window_size, stride=1, padding=pad)
    target_mean = F.avg_pool2d(target, kernel_size=window_size, stride=1, padding=pad)

    lmse = F.mse_loss(input_mean, target_mean)
    return lmse.item()

def ncc(input, target):
    """
    Compute Normalized Cross Correlation (NCC).
    :param input: Predicted image tensor [1, C, H, W]
    :param target: Ground Truth image tensor [1, C, H, W]
    """
    input = input - input.mean()
    target = target - target.mean()

    numerator = torch.sum(input * target)
    denominator = torch.sqrt(torch.sum(input ** 2) * torch.sum(target ** 2) + 1e-8)
    ncc = numerator / denominator
    return ncc.item()

def psnr(input, target):
    mse = F.mse_loss(input, target)
    return 10 * torch.log10(1.0 / mse)

def perceptual(vgg, input, target):
    input_features = vgg(input)
    target_features = vgg(target)
    return F.l1_loss(input_features, target_features)

def compute(model, inputs, GT, save_path, output_T=True):
    """Compute metrics and save the output image."""
    inputs, GT = inputs.to(device), GT.to(device)
    c = torch.tensor([0.5]).to(device).expand(inputs.size(0))

    with torch.no_grad():
        if output_T:
            # Forward pass for T_student
            _, _, _, _, _, _, _, final_out_T = model(inputs, c)
            pre_T = final_out_T[:, :3, :, :]

            # Metrics
            t_ssim = ssim(pre_T, GT).item()
            t_psnr = psnr(pre_T, GT).item()
            t_perceptual = perceptual(vgg16, pre_T, GT).item()
            t_lpips = lpips(pre_T, GT).mean().item()
            t_lmse = lmse(pre_T, GT)
            t_ncc = ncc(pre_T, GT)

            # Save output image
            output_image_T = transforms.ToPILImage()(pre_T.squeeze().cpu())
            output_image_T.save(f"{save_path}_T.jpg")

            return t_ssim, t_psnr, t_perceptual, t_lpips, t_lmse, t_ncc
        else:
            # Forward pass for R_student
            _, _, _, _, _, _, _, final_out_R = model(inputs, c)
            pre_R = final_out_R[:, :3, :, :]

            # Metrics
            r_ssim = ssim(pre_R, GT).item()
            r_psnr = psnr(pre_R, GT).item()
            r_perceptual = perceptual(vgg16, pre_R, GT).item()
            r_lpips = lpips(pre_R, GT).mean().item()
            r_lmse = lmse(pre_R, GT)
            r_ncc = ncc(pre_R, GT)

            # Save output image
            output_image_R = transforms.ToPILImage()(pre_R.squeeze().cpu())
            output_image_R.save(f"{save_path}_R.jpg")

            return r_ssim, r_psnr, r_perceptual, r_lpips, r_lmse, r_ncc

results = []
for image_path in tqdm(test_images, desc="Evaluating Test Images"):
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0)
    GT_tensor = input_tensor.clone()
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    save_path_t = os.path.join(output_test_dir, f"{image_name}_student")
    t_ssim, t_psnr, t_perceptual, t_lpips, t_lmse, t_ncc = compute(t_model, input_tensor, GT_tensor, save_path_t, output_T=True)

    save_path_r = os.path.join(output_test_dir, f"{image_name}_student")
    r_ssim, r_psnr, r_perceptual, r_lpips, r_lmse, r_ncc = compute( r_model, input_tensor, GT_tensor, save_path_r, output_T=False)


    results.append({
        'Image': image_name,
        'Tstu_SSIM': t_ssim, 'Tstu_PSNR': t_psnr,
        'Tstu_Perce': t_perceptual, 'Tstu_LPIPS': t_lpips, # Perceptual
        'Tstu_LMSE': t_lmse, 'Tstu_NCC': t_ncc,
        'Rstu_SSIM': r_ssim, 'Rstu_PSNR': r_psnr,
        'Rstu_Perce': r_perceptual, 'Rstu_LPIPS': r_lpips,
        'Rstu_LMSE': r_lmse, 'Rstu_NCC': r_ncc,
    })


df = pd.DataFrame(results)
df.to_excel(output_excel_path, index=False)
print(f"Results saved to {output_excel_path}")
##