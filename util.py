import torch
import math
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
import cv2


def vis_img(np_img):
    np_img = np.transpose(np_img, (1, 2, 0))
    plt.imshow(np_img)
    plt.show()


def load_img(path):
    img = misc.imread(path)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    img.unsqueeze_(0)
    return img.float()


def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return data2D


def warp(img2, flow):
    rec_img1 = np.zeros(img2.shape, dtype=img2.dtype)

    w, h = img2.shape[0:2]
    for i in range(w):
        for j in range(h):
            i2, j2 = (i, j) + flow[i][j]
            floor_i2 = int(math.floor(i2))
            floor_j2 = int(math.floor(j2))
            ceil_i2 = int(math.ceil(i2))
            ceil_j2 = int(math.ceil(j2))

            if not ((floor_i2 >= 0) and (floor_j2 >= 0) and (ceil_i2 < w) and (ceil_j2 < h)):
                for c in range(3):
                    rec_img1[i][j][c] = 0.0
                continue

            theta_x = i2 - floor_i2
            theta_y = j2 - floor_j2
            for c in range(3):
                rec_img1[i][j][c] = (1 - theta_x) * (1 - theta_y) * img2[floor_i2][floor_j2][c] + \
                                 theta_x * (1 - theta_y) * img2[ceil_i2][floor_j2][c] + \
                                 (1 - theta_x) * theta_y * img2[floor_i2][ceil_j2][c] + \
                                 theta_x * theta_y * img2[ceil_i2][ceil_j2][c]

    return rec_img1


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')

def show_cv2(img):
    npimg = img.numpy()
    cv2.imshow('img1', np.transpose(npimg, (1,2,0)))
    cv2.waitKey()
