import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from random import random
import time
import scipy as sp
import scipy.sparse.linalg
from typing import List, Tuple
import random


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

    # 将不属于mask=1的部分置为背景图片
    img_t_out[mask == 0] = img_t[mask == 0]
    return img_t_out  # np.clip(img_t_out, 0, 1)


# 计算放置obj梯度变化
def calculate_hue_variation(bg_hsv: np.ndarray, obj_hsv: np.ndarray, obj_h: int = 0, obj_w: int = 0, aa: int = 0,
                            bb: int = 0) -> float:
    """
        Returns a mixed gradient blended image with masked img_s over the img_t.
        :param bg_hsv: bg的hsv图片
        :param obj_hsv: obj的hsv图片
        :param obj_h: obj图片高
        :param obj_w: obj图片宽
        :param aa: obj图片在y轴的偏移量
        :param bb: obj图片在x轴的偏移量
        """
    # 获取对象图像的尺寸
    obj_height, obj_width = obj_h, obj_w

    # 计算对象放置的位置
    bg_h, bg_w = bg_hsv.shape[:2]
    x_offset = (bg_w - obj_img_w) // 2
    y_offset = (bg_h - obj_img_h) // 2

    # 提取重叠区域的Hue通道
    candidate = bg_hsv[y_offset + aa:y_offset + obj_height + aa, x_offset + bb:x_offset + obj_width + bb, 0]  # Hue通道

    # 计算变化量
    overlap_hue = candidate[obj_hsv[:, :, 0] > 0]  # 假设obj_hsv中的Hue通道大于0表示对象存在
    if overlap_hue.size > 0:
        hue_variation = np.std(overlap_hue) / (obj_height * obj_width)  # 归一化处理
        hue_variation = hue_variation * 100000
        print(f'Hue variation : {hue_variation}')
        return hue_variation

    return float('inf')


# 根据输入梯度寻找对应的放置区域
def find_best_rec(target_gradient: float, bg_hsv: np.ndarray, obj_hsv: np.ndarray) -> Tuple[int, int]:
    """
    Finds the best rectangle placement area based on a target gradient value.
    :param target_gradient: 目标梯度
    :param bg_hsv
    :param obj_hsv
    :return: aa, bb 代表了y,x的偏移量
    """
    bg_h, bg_w = bg_hsv.shape[:2]
    obj_height, obj_width = obj_hsv.shape[:2]

    # 计算对象放置的中心位置
    y_center = (bg_h - obj_height) // 2
    x_center = (bg_w - obj_width) // 2
    step = 10
    # 存储所有满足条件的位置
    valid_positions = []

    # 遍历整个背景图像，寻找所有满足条件的矩形区域
    for y in range(bg_h - obj_height + 1):
        for x in range(bg_w - obj_width + 1):
            # 按照step步长滑动
            if y % step != 0 or x % step != 0:
                continue
            # 提取重叠区域的Hue通道
            candidate_hue = bg_hsv[y:y + obj_height, x:x + obj_width, 0]  # Hue通道
            candidate_value = bg_hsv[y:y + obj_height, x:x + obj_width, 2]  # Value通道

            # 计算变化量
            overlap_hue = candidate_hue[obj_hsv[:, :, 0] > 0]  # 假设obj_hsv中的Hue通道大于0表示对象存在
            # 计算明度通道的变化量
            value_channel = candidate_value[obj_hsv[:, :, 2] > 0]  # 获取明度通道数据
            if value_channel.size > 0:
                value_variation = np.std(value_channel)  # 计算明度通道的标准差
                value_variation = value_variation * 10  # 与色调变化量使用相同的缩放因子
            else:
                value_variation = 0

            if overlap_hue.size > 0:
                hue_variation = np.std(overlap_hue)  # 不归一化，直接使用标准差
                hue_variation = hue_variation * 10
                # print(f'Now Hue variation is: {hue_variation},keep detecting')
                total_variation = 0.5 * hue_variation + 0.5 * value_variation
                print(f'Now total variation is: {total_variation},keep detecting')
                # 检查是否在目标梯度值的范围内
                if target_gradient - 0.01 <= total_variation <= target_gradient + 0.01:
                    # 计算偏移量
                    aa = y - y_center
                    bb = x - x_center
                    print(f'Correct! Now aa is: {aa},bb is: {bb}')
                    valid_positions.append((aa, bb))  # 将满足条件的位置添加到列表中

    # 如果找到了满足条件的位置，随机选择一个返回
    if valid_positions:
        print(f'Now valid_positions is: {valid_positions}')
        # 随机选取一组aa,bb，满足绝对值均小于160
        valid_positions = [pos for pos in valid_positions if abs(pos[0]) < 160 and abs(pos[1]) < 160]
        print(f'Then valid_positions is: {valid_positions}')
        if valid_positions:
            return random.choice(valid_positions)

    # 如果没有找到合适的矩形，返回默认值
    return 0, 0


# 主函数部分
# 创建保存结果的文件夹
if not os.path.exists('small_data_gun'):
    os.makedirs('small_data_gun')

# 读取obj_img
obj_img = cv2.imread('samples/scissor.JPEG')
hsv_obj = cv2.cvtColor(obj_img, cv2.COLOR_BGR2HSV)
# 定义黑色背景的颜色范围
lower_white = np.array([0, 0, 240])  # 白色的HSV范围
upper_white = np.array([180, 20, 255])
mask = cv2.inRange(hsv_obj, lower_white, upper_white)
# 对掩码进行形态学操作，去除噪点
kernel = np.ones((3,3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask_inv = cv2.bitwise_not(mask)
result = cv2.bitwise_and(obj_img, obj_img, mask=mask_inv)
# 将背景区域设置为白色
background = np.full_like(obj_img, [255, 255, 255])  # 创建白色背景
# 修改替换逻辑：只在mask为255(白色背景)的位置替换为白色背景
obj_img = cv2.bitwise_and(obj_img, obj_img, mask=mask_inv) + cv2.bitwise_and(background, background, mask=mask)
# show_images([cv2.cvtColor(obj_img, cv2.COLOR_BGR2RGB)],["No Black Background"])
# show_images([cv2.cvtColor(obj_img, cv2.COLOR_BGR2HSV)],["HSV Object image"])
save_path = 'samples/final_image.jpg'
cv2.imwrite(save_path, obj_img)
obj_img = get_image(save_path)

# 遍历samples/small_data下的所有图片
for bg_file in os.listdir('samples/small_data'):
    try:
        bg_img = get_image(os.path.join('samples/small_data', bg_file))
        
        bg_img_h, bg_img_w = bg_img.shape[:2]
        obj_img_h, obj_img_w = obj_img.shape[:2]

        x_offset = (bg_img_w - obj_img_w) // 2
        y_offset = (bg_img_h - obj_img_h) // 2

        new_obj_img = np.ones_like(bg_img)
        new_mask_img = np.zeros_like(bg_img[:, :, 0])
        
        mask_y, mask_x = np.where(mask == 255)
        aa = -50
        bb = -50
        calculate_hue_variation(bg_img, obj_img, obj_img_h, obj_img_w, aa, bb)
        aap, bbp = find_best_rec(2.5, bg_img, obj_img) # 调整梯度大小
        print(f'Finally, aap is: {aap},bbp is: {bbp}')
        aa = aap
        bb = bbp

        new_obj_img[y_offset + aa:y_offset + obj_img_h + aa, x_offset + bb:x_offset + obj_img_w + bb] = obj_img
        new_mask_img[y_offset + aa:y_offset + obj_img_h + aa, x_offset + bb:x_offset + obj_img_w + bb] = 1
        for y, x in zip(mask_y, mask_x):
            new_mask_img[y_offset + y + aa, x_offset + x + bb] = 0

        mix_img = np.zeros(bg_img.shape)
        # show_images(
        #     [bg_img, new_obj_img, new_mask_img, mix_img],
        #     ["Background image", "Object image", "Mask image", "Blended image"]
        # )

        kernel = np.ones((3, 3), np.uint8)
        dilated_img = cv2.dilate(new_obj_img, kernel, iterations=1)
        eroded_img = cv2.erode(dilated_img, kernel, iterations=1)
        new_obj_img = cv2.cvtColor(eroded_img.astype('float32'), cv2.COLOR_BGR2HSV)

        bg_img = cv2.cvtColor(bg_img.astype('float32'), cv2.COLOR_BGR2HSV)

        kernel = np.ones((15, 15), np.uint8)
        dilated_mask = cv2.dilate(new_mask_img, kernel, iterations=1)
        blurred_mask = cv2.GaussianBlur(dilated_mask, (15, 15), 0)
        binary_blurred_mask = (blurred_mask > 0.5).astype(np.uint8)

        for b in np.arange(1, 3):
            mix_img[:, :, b] = mixed_blend(new_obj_img[:, :, b], binary_blurred_mask, bg_img[:, :, b].copy())

        mix_img[:, :, 0] = bg_img[:, :, 0].copy()
        valid_region = binary_blurred_mask > 0
        mix_img[valid_region, 0] = np.minimum(bg_img[valid_region, 0], new_obj_img[valid_region, 0])

        # 创建一个扩展的掩膜区域
        expanded_mask_y = []
        expanded_mask_x = []
        for y, x in zip(mask_y, mask_x):
            # 向四周扩展5个像素
            for dy in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
                for dx in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
                    expanded_mask_y.append(y + dy)
                    expanded_mask_x.append(x + dx)
        # 应用扩展后的掩膜
        for y, x in zip(expanded_mask_y, expanded_mask_x):
            mix_img[y_offset + y + aa, x_offset + x + bb, 0] = bg_img[
                y_offset + y + aa, x_offset + x + bb, 0]  # 使用背景的色调
        mix_img = cv2.cvtColor(mix_img.astype('float32'), cv2.COLOR_HSV2BGR)
        
        # 确保mix_img的值在0-1范围内
        mix_img = np.clip(mix_img, 0, 1)

        # 使用plt保存mix_img图片
        output_filename = os.path.join('small_data_gun', f'result_{bg_file}')
        plt.imsave(output_filename, mix_img)
        
    except Exception as e:
        print(f'Error processing {bg_file}: {str(e)}')
        continue


