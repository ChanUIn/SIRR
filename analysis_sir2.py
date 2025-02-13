import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

input_dir = '../cvpr21-absorption/data2-glass/'
filename = os.listdir(os.path.join(input_dir, '3', 't'))
ks = 50
f = np.ones((ks+1, ks+1)) / (ks+1) / (ks+1)
k = 0
tx = 500
th = 10
th2 = 0.0
tt = 0.95
results = []

for i in range(1, 21):
    img3 = cv2.imread(os.path.join(input_dir, '3', 'syn', filename[i+2]), cv2.IMREAD_GRAYSCALE).astype(float)
    img5 = cv2.imread(os.path.join(input_dir, '5', 'syn', filename[i+2]), cv2.IMREAD_GRAYSCALE).astype(float)
    img10 = cv2.imread(os.path.join(input_dir, '10', 'syn', filename[i+2]), cv2.IMREAD_GRAYSCALE).astype(float)
    img3t = cv2.imread(os.path.join(input_dir, '3', 't', filename[i+2]), cv2.IMREAD_GRAYSCALE).astype(float)
    img5t = cv2.imread(os.path.join(input_dir, '5', 't', filename[i+2]), cv2.IMREAD_GRAYSCALE).astype(float)
    img10t = cv2.imread(os.path.join(input_dir, '10', 't', filename[i+2]), cv2.IMREAD_GRAYSCALE).astype(float)

    img3r = cv2.imread(os.path.join(input_dir, '3', 'r', filename[i+2]), cv2.IMREAD_GRAYSCALE).astype(float)
    img5r = cv2.imread(os.path.join(input_dir, '5', 'r', filename[i+2]), cv2.IMREAD_GRAYSCALE).astype(float)
    img10r = cv2.imread(os.path.join(input_dir, '10', 'r', filename[i+2]), cv2.IMREAD_GRAYSCALE).astype(float)

    h, w = img3.shape

    mssim = ssim(img3t.astype(np.uint8), img5t.astype(np.uint8))
    e35 = mssim
    mssim = ssim(img3t.astype(np.uint8), img10t.astype(np.uint8))
    e310 = mssim
    mssim = ssim(img5t.astype(np.uint8), img10t.astype(np.uint8))
    e510 = mssim

    c3 = cv2.filter2D(img3r, -1, f, borderType=cv2.BORDER_CONSTANT)
    c5 = cv2.filter2D(img3r, -1, f, borderType=cv2.BORDER_CONSTANT)
    c10 = cv2.filter2D(img10r, -1, f, borderType=cv2.BORDER_CONSTANT)

    if e35 > tt:
        cc = c3 + c5

        index_min = np.argmin(cc)
        index_i = np.ceil(index_min / (h - ks))
        index_j = index_min - (index_i - 1) * (h - ks)
        a = img3[int(index_j):int(index_j) + ks, int(index_i):int(index_i) + ks]
        b = img5[int(index_j):int(index_j) + ks, int(index_i):int(index_i) + ks]
        at = img3t[int(index_j):int(index_j) + ks, int(index_i):int(index_i) + ks]
        bt = img3t[int(index_j):int(index_j) + ks, int(index_i):int(index_i) + ks]
        ar = img3r[int(index_j):int(index_j) + ks, int(index_i):int(index_i) + ks]
        br = img5r[int(index_j):int(index_j) + ks, int(index_i):int(index_i) + ks]

        ra = (a - np.mean(a)) / (at - np.mean(at))
        rb = (b - np.mean(b)) / (bt - np.mean(bt))

        ma = ((ra < 1) & (ra > th2) & (ar < th)).astype(float)
        mb = ((rb < 1) & (rb > th2) & (br < th)).astype(float)

        if np.sum(ma) > tx and np.sum(mb) > tx:
            k += 1
            label = [i, 3, 5]
            result = [np.sum(ra * ma) / np.sum(ma), np.sum(rb * mb) / np.sum(mb)]
            results.append(result)

    if e310 > tt:
        cc = c3 + c10

        index_min = np.argmin(cc)
        index_i = np.ceil(index_min / (h - ks))
        index_j = index_min - (index_i - 1) * (h - ks)
        a = img3[int(index_j):int(index_j) + ks, int(index_i):int(index_i) + ks]
        b = img10[int(index_j):int(index_j) + ks, int(index_i):int(index_i) + ks]
        at = img3t[int(index_j):int(index_j) + ks, int(index_i):int(index_i) + ks]
        bt = img3t[int(index_j):int(index_j) + ks, int(index_i):int(index_i) + ks]
        ar = img3r[int(index_j):int(index_j) + ks, int(index_i):int(index_i) + ks]
        br = img10r[int(index_j):int(index_j) + ks, int(index_i):int(index_i) + ks]

        ra = (a - np.mean(a)) / (at - np.mean(at))
        rb = (b - np.mean(b)) / (bt - np.mean(bt))

        ma = ((ra < 1) & (ra > th2) & (ar < th)).astype(float)
        mb = ((rb < 1) & (rb > th2) & (br < th)).astype(float)

        if np.sum(ma) > tx and np.sum(mb) > tx:
            k += 1
            label = [i, 3, 10]
            result = [np.sum(ra * ma) / np.sum(ma), np.sum(rb * mb) / np.sum(mb)]
            results.append(result)

    if e510 > tt:
        cc = c5 + c10

        index_min = np.argmin(cc)
        index_i = np.ceil(index_min / (h - ks))
        index_j = index_min - (index_i - 1) * (h - ks)
        a = img5[int(index_j):int(index_j) + ks, int(index_i):int(index_i) + ks]
        b = img10[int(index_j):int(index_j) + ks, int(index_i):int(index_i) + ks]
        at = img5t[int(index_j):int(index_j) + ks, int(index_i):int(index_i) + ks]
        bt = img5
