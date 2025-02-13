import numpy as np
from scipy.ndimage import convolve
from skimage import io, color, exposure
from scipy.io import loadmat

def ssim_modify(img1, img2, K=[0.01, 0.03], window_size=11, L=255):
    def fspecial_gaussian(shape=(3, 3), sigma=0.5):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1, -n:n+1]
        h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def filter2(b, x):
        return convolve(x, b, mode='constant', cval=0.0)

    def imfilter(f, h, boundary='symm'):
        return convolve(f, h[::-1, ::-1], mode='constant', cval=0.0)

    if len(img1.shape) == 3:
        img1 = color.rgb2gray(img1)
    if len(img2.shape) == 3:
        img2 = color.rgb2gray(img2)

    if img1.shape != img2.shape:
        return -np.inf, -np.inf, -np.inf, -np.inf

    M, N = img1.shape

    if (M < 11) or (N < 11):
        return -np.inf, -np.inf, -np.inf, -np.inf

    if window_size % 2 == 0:
        window_size += 1

    window = fspecial_gaussian((window_size, window_size), sigma=1.5)
    window /= np.sum(window)

    img1 = img1.astype(float)
    img2 = img2.astype(float)

    f = max(1, round(min(M, N) / 256))
    if f > 1:
        lpf = np.ones((f, f)) / (f * f)
        img1 = imfilter(img1, lpf)
        img2 = imfilter(img2, lpf)

        img1 = img1[::f, ::f]
        img2 = img2[::f, ::f]

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    mu1 = filter2(window, img1, boundary='symm')
    mu2 = filter2(window, img2, boundary='symm')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter2(window, img1 * img1, boundary='symm') - mu1_sq
    sigma2_sq = filter2(window, img2 * img2, boundary='symm') - mu2_sq
    sigma12 = filter2(window, img1 * img2, boundary='symm') - mu1_mu2

    si_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    sip_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    ssim_map = si_map * sip_map

    if C1 > 0 and C2 > 0:
        ssim_map *= si_map * sip_map
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        ssim_map = np.ones_like(mu1)
        index = (denominator1 * denominator2 > 0)
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    mssim = np.mean(ssim_map)
    msi = np.mean(si_map)
    msip = np.mean(sip_map)

    return mssim, msi, msip, ssim_map

# Example usage:
img1 = io.imread('path_to_image1.jpg')
img2 = io.imread('path_to_image2.jpg')

mssim, msi, msip, ssim_map = ssim_modify(img1, img2)
print("MSSIM:", mssim)
print("MSI:", msi)
print("MSIP:", msip)
