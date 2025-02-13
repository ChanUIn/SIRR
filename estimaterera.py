import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from Get_Map import get_map


GT_dir = r'E:/CHAN/absorption_2/main/matlab/dataset/training_data/8700_540/real_word/ret/'
mix_dir = r'E:/CHAN/absorption_2/main/matlab/dataset/training_data/8700_540/real_word/reI/'
output_dir = r'E:/CHAN/absorption_2/main/matlab/dataset/training_data/8700_540/real_word/'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, '1216estimate.txt')

def monte_carlo(h, w, num_rays=1000, max_depth=5):
    """
        執行Monte Carlo光線追蹤來計算反射和透射圖。

        Args:
            h (int): Height of the output map.
            w (int): Width of the output map.
            num_rays (int): Number of rays to trace per pixel.
            max_depth (int): Maximum recursion depth for ray tracing.

        Returns:
            tuple: Reflection (R_map) and Transmission (T_map) maps.
        """
    R_map = np.zeros((h, w), dtype=np.float32)
    T_map = np.zeros((h, w), dtype=np.float32)
    # 迭代每個像素
    for y in tqdm(range(h), desc="Tracing rays"):
        for x in range(w):
            R_total, T_total = 0, 0

            for _ in range(num_rays):
                theta = np.random.uniform(-np.pi / 4, np.pi / 4)
                phi = np.random.uniform(0, 2 * np.pi)
                direction = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])

                R, T = trace_ray(direction, max_depth)
                R_total += R
                T_total += T

            R_map[y, x] = R_total / num_rays
            T_map[y, x] = T_total / num_rays

    return T_map, R_map

def trace_ray(direction, max_depth, current_depth=0, n1=1.0, n2=1.474):
    """
    使用Monte Carlo模擬追蹤單條光線。

    Args:
        direction (np.array): Direction of the ray.
        max_depth (int): Maximum recursion depth.
        current_depth (int): Current recursion depth.
        n1 (float): Refractive index of the first medium.
        n2 (float): Refractive index of the second medium.

    Returns:
        tuple: Reflection and transmission contributions for the ray.
    """
    if current_depth >= max_depth:
        return 0, 1

    theta = np.arccos(direction[2])
    Rs = ((n1 * np.cos(theta) - n2 * np.sqrt(1 - (n1 / n2 * np.sin(theta)) ** 2)) /
          (n1 * np.cos(theta) + n2 * np.sqrt(1 - (n1 / n2 * np.sin(theta)) ** 2))) ** 2
    Rp = ((n2 * np.cos(theta) - n1 * np.sqrt(1 - (n1 / n2 * np.sin(theta)) ** 2)) /
          (n2 * np.cos(theta) + n1 * np.sqrt(1 - (n1 / n2 * np.sin(theta)) ** 2))) ** 2
    R_prob = 0.5 * (Rs + Rp)
    T_prob = 1 - R_prob

    if np.random.rand() < R_prob:
        reflected_dir = direction - 2 * np.dot(direction, [0, 0, 1]) * [0, 0, 1]
        return trace_ray(reflected_dir, max_depth, current_depth + 1, n1, n2)
    else:
        transmitted_dir = direction
        return trace_ray(transmitted_dir, max_depth, current_depth + 1, n2, n1)


def get_map(h, w):
    FoV_list = [19.45, 23.12, 30.08, 33.40, 39.60, 48.46, 65.47, 73.74]
    thick_list = np.array([3, 6, 9]) * 0.001

    while True:
        g_dis = 0
        while g_dis < 0.2:
            g_dis = np.random.rand(1) * (5 - 0.2) + 0.2
        g_size = 0
        while g_size < 0.4:
            g_size = np.random.rand(1) * (3 - 0.4) + 0.4
        g_angle1 = 90
        while np.abs(g_angle1) > 60:
            g_angle1 = 10 * (np.random.randn(1, 1))
        g_angle2 = 90
        while np.abs(g_angle2) > 15:
            g_angle2 = 4 * (np.random.randn(1, 1))
        FoV = np.random.rand(1) * (73.74 - 19.45) + 19.45

        thickness = np.random.rand(1) * 0.007 + 0.003

        cor_x2 = np.tan((-FoV / 2 + np.abs(g_angle1)) / 180 * np.pi) * g_dis
        cor_x3 = np.tan((FoV / 2 + np.abs(g_angle1)) / 180 * np.pi) * g_dis

        if cor_x3 - cor_x2 < g_size:
            break

    n2 = 1.474

    map_T = compute_theta_map(w, h, g_dis, g_angle1, g_angle2, FoV, g_size)
    [T, R] = compute_map(map_T, n2)

    cos_map_T2 = 1.0 / np.sqrt(1 - (1 / n2 * np.sin(map_T)) ** 2)
    k_c = np.random.rand(1) * (32 - 4) + 4
    alpha = np.exp(-k_c * thickness * cos_map_T2)

    coe_R = R + R * (T * R * alpha * alpha) / (
            1 - R * R * alpha * alpha)
    coe_T = (T * T * alpha) / (1 - R * R * alpha * alpha)

    return coe_T, coe_R

def compute_theta_map(w, h, g_dis, angle1, angle2, FoV, g_size):
    cor_x2 = np.tan((-FoV/2 + np.abs(angle1))/180*np.pi)*g_dis
    cor_x3 = np.tan((FoV/2 + np.abs(angle1))/180*np.pi)*g_dis
    centerX = 0
    centerY = 0
    centerZ = w/2/np.tan(FoV/2/180*np.pi)
    if cor_x3-cor_x2 > g_size:
        step = g_size*np.cos(angle1/180*np.pi)/w
    else:
        step = 1
    map_T = np.zeros((h, w))
    n = np.array([0.0, 0.0, 0.0], dtype=float)  # Ensure n is of float type
    nv = np.array([np.sin(angle1/180*np.pi)*np.cos(angle2/180*np.pi),
                   np.sin(angle1/180*np.pi)*np.sin(angle2/180*np.pi),
                   np.cos(angle1/180*np.pi)])
    for i in range(-w//2, w//2):
        for j in range(-h//2, h//2):
            n[0] = (i+0.5)*step+centerX
            n[1] = (j+0.5)*step+centerY
            n[2] = centerZ
            n /= np.linalg.norm(n)
            map_T[j+h//2, i+w//2] = np.arccos(np.dot(nv.flatten(), n))

    return map_T

def compute_map(map_T, n2, wavelength=550e-9):
    """
        計算給定入射角(map_T) 的反射(R)和透射(T)係數。

        Args:
            map_T (numpy.ndarray): Angle of incidence map.
            n2 (float): Refractive index of the second medium.
            wavelength (float): Wavelength of incident light in meters.

        Returns:
            tuple: Transmission (T) and reflection (R) maps.
    """
    n1 = 1

    def Rs1(theta, n1, n2):
        cos_theta_t = np.sqrt(1 - (n1 / n2 * np.sin(theta)) ** 2)
        return ((n1 * np.cos(theta) - n2 * cos_theta_t) /
                (n1 * np.cos(theta) + n2 * cos_theta_t)) ** 2

    def Rp1(theta, n1, n2):
        cos_theta_t = np.sqrt(1 - (n1 / n2 * np.sin(theta)) ** 2)
        return ((n2 * np.cos(theta) - n1 * cos_theta_t) /
                (n2 * np.cos(theta) + n1 * cos_theta_t)) ** 2

    R = 0.5 * (Rs1(map_T, n1, n2) + Rp1(map_T, n1, n2))
    T = 1 - R

    n2_dispersion = n2 + 0.02 * np.log(wavelength / 550e-9)
    R *= n2_dispersion
    T *= n2_dispersion

    return T, R

with open(output_file, 'w') as f:
    count = 13701  # start number
    gt_filelist = sorted([file for file in os.listdir(GT_dir) if file.lower().endswith('.jpg')])
    mix_filelist = sorted([file for file in os.listdir(mix_dir) if file.lower().endswith('.jpg')])

    for gt_filename, mix_filename in tqdm(zip(gt_filelist, mix_filelist), total=len(gt_filelist), desc="Processing images"):
        gt_image = cv2.imread(os.path.join(GT_dir, gt_filename))
        mix_image = cv2.imread(os.path.join(mix_dir, mix_filename))

        if gt_image is not None and mix_image is not None:
            coe_T, coe_R = get_map(gt_image.shape[0], gt_image.shape[1])
            R = coe_R.mean()

            f.write(f'{count:05d}.jpg\t{coe_T.mean()}\t{R}\n')
            count += 1

print(f'File {output_file} generated successfully.')

