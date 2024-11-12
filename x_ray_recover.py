import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from random import random
import time
import scipy as sp
import scipy.sparse.linalg
from typing import List, Tuple


def get_image(img_path: str, mask: bool = False, scale: bool = True) -> np.array:
    """
    Gets image in appriopiate format

    Parameters:
    img_path (str): Image path
    mask (bool): True if read mask image
    scale (bool): True if read and scale image to 0-1

    Returns:
    np.array: Image in numpy array
    """
    if mask:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return np.where(binary_mask == 255, 1, 0)

    if scale:
        return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype('double') / 255.0

    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)


def show_images(
        imgs: List[np.array], titles: List[str], figsize: Tuple[int] = (15, 10)
) -> None:
    """
    Show images with tites

    Parameters:
    imgs (List): List of images
    titles (List): List of titles
    figsize (Tuple): Figure size
    """
    idx = 1
    fig = plt.figure(figsize=figsize)

    for img, title in zip(imgs, titles):
        ax = fig.add_subplot(1, len(imgs), idx)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(title)
        idx += 1
    plt.show()


def neighbours(i: int, j: int, max_i: int, max_j: int) -> List[Tuple[int, int]]:
    """
    Returns 4-connected neighbours for given pixel point.
    :param i: i-th index position
    :param j: j-th index position
    :param max_i: max possible i-th index position
    :param max_j: max possible j-th index position
    """
    pairs = []

    for n in [-1, 1]:
        if 0 <= i + n <= max_i:
            pairs.append((i + n, j))
        if 0 <= j + n <= max_j:
            pairs.append((i, j + n))

    return pairs


def mixed_blend(
        img_s: np.ndarray,
        mask: np.ndarray,
        img_t: np.ndarray
) -> np.ndarray:
    """
    Returns a mixed gradient blended image with masked img_s over the img_t.
    :param img_s: the image containing the foreground object
    :param mask: the mask of the foreground object in img_s
    :param img_t: the background image
    """
    img_s_h, img_s_w = img_s.shape

    nnz = (mask > 0).sum()
    im2var = -np.ones(mask.shape[0:2], dtype='int32')
    im2var[mask > 0] = np.arange(nnz)

    ys, xs = np.where(mask == 1)

    A = sp.sparse.lil_matrix((4 * nnz, nnz))
    b = np.zeros(4 * nnz)

    e = 0
    for n in range(nnz):
        y, x = ys[n], xs[n]

        for n_y, n_x in neighbours(y, x, img_s_h - 1, img_s_w - 1):
            ds = img_s[y][x] - img_s[n_y][n_x]
            dt = img_t[y][x] - img_t[n_y][n_x]
            d = ds if abs(ds) > abs(dt) else dt

            A[e, im2var[y][x]] = 1
            b[e] = d

            if im2var[n_y][n_x] != -1:
                A[e, im2var[n_y][n_x]] = -1
            else:
                b[e] += img_t[n_y][n_x]
            e += 1

    A = sp.sparse.csr_matrix(A)
    v = sp.sparse.linalg.lsqr(A, b)[0]

    img_t_out = img_t.copy()

    for n in range(nnz):
        y, x = ys[n], xs[n]
        img_t_out[y][x] = v[im2var[y][x]]

    return img_t_out  # np.clip(img_t_out, 0, 1)

# 主函数部分
bg_img = get_image('samples/BackGround1.jpg')

# obj_img = get_image('samples/bottle1.jpg')

obj_img = cv2.imread('samples/bottle1.jpg')
hsv_obj = cv2.cvtColor(obj_img, cv2.COLOR_BGR2HSV)
# 定义黑色背景的颜色范围
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])
mask = cv2.inRange(hsv_obj, lower_black, upper_black)
mask_inv = cv2.bitwise_not(mask)
result = cv2.bitwise_and(obj_img, obj_img, mask=mask_inv)
# 将黑色背景区域设置为白色
background = np.full_like(obj_img, [255, 255, 255])
obj_img = np.where(result == 0, background, result)
show_images([cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)],["No Black Background"])

save_path = 'samples/final_image.jpg'
cv2.imwrite(save_path, obj_img)
obj_img = get_image(save_path)

# mask_img =  get_image('samples/gun2.jpg', mask=True)
bg_img = cv2.blur(bg_img, (5,5))
obj_img = cv2.blur(obj_img, (5,5))

bg_img_h, bg_img_w = bg_img.shape[:2]
obj_img_h, obj_img_w = obj_img.shape[:2]

x_offset = (bg_img_w - obj_img_w)//2
y_offset = (bg_img_h - obj_img_h)//2

new_obj_img = np.ones_like(bg_img)
new_obj_img[y_offset:y_offset+obj_img_h, x_offset:x_offset+obj_img_w] = obj_img

new_mask_img = np.zeros_like(bg_img[:,:,0])
print(new_mask_img[0:5, 0:5])
new_mask_img[y_offset:y_offset+obj_img_h, x_offset:x_offset+obj_img_w] = 1
mask_y, mask_x = np.where(mask == 255)
aa=0
bb=0
# tmp_img = cv2.imread('samples/final_image.jpg')
# gray_obj = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray_obj, 50, 150)
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# new_mask_img = np.zeros_like(gray_obj)
# cv2.drawContours(new_mask_img, contours, -1, color=255, thickness=cv2.FILLED)
# # 反转掩码，使掩码外的区域为黑色，物体为白色
# new_mask_img = cv2.bitwise_not(new_mask_img)

print(bg_img.shape, new_obj_img.shape, new_mask_img.shape)
mix_img = np.zeros(bg_img.shape)
show_images(
    [bg_img, new_obj_img, new_mask_img, mix_img],
    ["Background image", "Object image", "Mask image", "Blended image"]
)

new_obj_img = cv2.cvtColor(new_obj_img.astype('float32'), cv2.COLOR_BGR2HSV)

# shouldn't convert mask into hsv format
bg_img = cv2.cvtColor(bg_img.astype('float32'), cv2.COLOR_BGR2HSV)


for b in np.arange(1,3): # don't use the Hue Channel
    mix_img[:,:,b] = mixed_blend(new_obj_img[:,:,b], new_mask_img, bg_img[:,:,b].copy())

mix_img[:,:, 0] = np.maximum(bg_img[:,:,0], new_obj_img[:,:,0])
show_images(
    [cv2.cvtColor(bg_img.astype('float32'), cv2.COLOR_HSV2BGR),   cv2.cvtColor(mix_img.astype('float32'), cv2.COLOR_HSV2BGR)],
    ["Background image", "Blended image"]
)

blue_threshold = 100
mix_img[y_offset:y_offset+obj_img_h, x_offset:x_offset+obj_img_w, 0] = np.where(
    new_obj_img[y_offset:y_offset+obj_img_h, x_offset:x_offset+obj_img_w, 0] > blue_threshold,
    bg_img[y_offset:y_offset+obj_img_h, x_offset:x_offset+obj_img_w, 0],
    np.minimum(bg_img[y_offset:y_offset+obj_img_h, x_offset:x_offset+obj_img_w, 0],
               new_obj_img[y_offset:y_offset+obj_img_h, x_offset:x_offset+obj_img_w, 0])
)
# mix_img[y_offset:y_offset+obj_img_h, x_offset:x_offset+obj_img_w, 0] \
#     = np.minimum(bg_img[y_offset:y_offset+obj_img_h, x_offset:x_offset+obj_img_w, 0], new_obj_img[y_offset:y_offset+obj_img_h, x_offset:x_offset+obj_img_w, 0])

mix_img = cv2.cvtColor(mix_img.astype('float32'), cv2.COLOR_HSV2BGR)
bg_img = cv2.cvtColor(bg_img.astype('float32'), cv2.COLOR_HSV2BGR)

show_images(
    [bg_img,  mix_img],
    ["Background image", "Blended image"]
)

show_images(
    [bg_img, new_obj_img, new_mask_img, mix_img],
    ["Background image", "Object image", "Mask image", "Blended image"]
)
