import random

import numpy as np


def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    imgarr = np.asarray(img)
    proc_img = np.empty_like(imgarr, np.float32)

    proc_img[..., 0] = (imgarr[..., 0] - mean[0]) / std[0]
    proc_img[..., 1] = (imgarr[..., 1] - mean[1]) / std[1]
    proc_img[..., 2] = (imgarr[..., 2] - mean[2]) / std[2]
    return proc_img


def random_fliplr(img):
    if random.random() > 0.5:
        img = np.fliplr(img)

    return img


def random_flipud(img):
    if random.random() > 0.5:
        img = np.flipud(img)
    return img


def random_rot(img):
    k = random.randrange(3) + 1  # 1<=k<=4
    img = np.rot90(img, k).copy()
    return img


def encode_cmap(label):
    cmap = colormap()
    return cmap[label.astype(np.int16), :]


def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])
    cmap = cmap / 255 if normalized else cmap
    return cmap
