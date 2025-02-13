# TSmodel_test.py
import torch
import os
import pandas as pd
from TSmodel import T_teacher_net, R_teacher_net
import pytorch_ssim
import lpips
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from student import psnr, compute_perceptual_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ssim_loss = pytorch_ssim.SSIM()
lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)

t_model_path = "E:/CHAN/absorption_2/main/GFLO70148_newmodel_13700_code_result_256/mix_data/Tmodels/T_teacher_epoch_200.pth"
r_model_path = "E:/CHAN/absorption_2/main/GFLO70148_newmodel_13700_code_result_256/mix_data/Tmodels/R_teacher_epoch_200.pth"
test_data_dir = "E:/CHAN/absorption_2/main/matlab/dataset/test/18/syn"
output_excel_path = "E:/CHAN/absorption_2/main/GFLO70148_newmodel_13700_code_result_256/onlyteacher0212.xlsx"
output_test_dir = "E:/CHAN/absorption_2/main/GFLO70148_newmodel_13700_code_result_256/onlyteacher0212"
os.makedirs(output_test_dir, exist_ok=True)

t_model = T_teacher_net().to(device)
r_model = R_teacher_net().to(device)
t_model.load_state_dict(torch.load(t_model_path))
r_model.load_state_dict(torch.load(r_model_path))

t_model.eval()
r_model.eval()

vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device)
for param in vgg16.parameters():
    param.requires_grad = False

transform = transforms.Compose([
    transforms.ToTensor()
])

test_images = [os.path.join(test_data_dir, img) for img in os.listdir(test_data_dir) if img.endswith(('.png', '.jpg'))]

def compute_psnr(input, target):
    """Compute PSNR."""
    mse = F.mse_loss(input, target)
    return 10 * torch.log10(1.0 / mse)

def compute_lmse(input, target, window_size=7):
    """Compute Local Mean Square Error (LMSE)."""
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

def compute_metrics_and_save_output(model, inputs, GT, save_path):
    inputs, GT = inputs.to(device), GT.to(device)

    with torch.no_grad():
        if isinstance(model, T_teacher_net):
            _, _, _, _, _, outputs = model(inputs, torch.tensor([0.5]).to(inputs.device).expand(inputs.size(0)))
        elif isinstance(model, R_teacher_net):
            _, _, _, outputs = model(inputs, torch.tensor([0.5]).to(inputs.device).expand(inputs.size(0)))
        else:
            raise ValueError("Unsupported model type")

        outputs_rgb = outputs[:, :3, :, :]
        ssim_value = ssim_loss(outputs_rgb, GT).item()
        psnr_value = compute_psnr(outputs_rgb, GT)
        perceptual_loss_value = compute_perceptual_loss(vgg16, outputs_rgb, GT).item()
        lpips_value = lpips_loss_fn(outputs_rgb, GT).mean().item()
        lmse_value = compute_lmse(outputs_rgb, GT)
        ncc_value = ncc(outputs_rgb, GT)

        output_image = transforms.ToPILImage()(outputs_rgb.squeeze().cpu())
        output_image.save(save_path)

    return ssim_value, psnr_value, perceptual_loss_value, lpips_value, lmse_value, ncc_value

results = []
for image_path in tqdm(test_images, desc="Evaluating Test Images"):
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0)
    GT_tensor = input_tensor.clone()
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    save_path_t = os.path.join(output_test_dir, f"{image_name}_T_teacher.jpg")
    t_ssim, t_psnr, t_perceptual, t_lpips, t_lmse, t_ncc = compute_metrics_and_save_output(t_model, input_tensor, GT_tensor, save_path_t)

    save_path_r = os.path.join(output_test_dir, f"{image_name}_R_teacher.jpg")
    r_ssim, r_psnr, r_perceptual, r_lpips, r_lmse, r_ncc = compute_metrics_and_save_output(r_model, input_tensor, GT_tensor, save_path_r)

    results.append({
        'Image': image_name,
        'T_SSIM': t_ssim, 'T_PSNR': float(t_psnr),
        'T_Perceptual': t_perceptual, 'T_LPIPS': t_lpips,
        'T_LMSE': t_lmse, 'T_NCC': t_ncc,
        'R_SSIM': r_ssim, 'R_PSNR': float(r_psnr),
        'R_Perceptual': r_perceptual, 'R_LPIPS': r_lpips,
        'R_LMSE': r_lmse, 'R_NCC': r_ncc,
    })

df = pd.DataFrame(results)
df.to_excel(output_excel_path, index=False)
print(f"Results saved to {output_excel_path}")
