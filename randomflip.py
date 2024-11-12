import cv2
import numpy as np
import random

def rotate_image_with_background(img):
    height, width = img.shape[:2]
    angle = random.uniform(-180, 180)
    # 计算旋转矩阵
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 计算旋转后的四个角点
    cos_angle = np.abs(rotation_matrix[0, 0])
    sin_angle = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin_angle) + (width * cos_angle))
    new_height = int((height * cos_angle) + (width * sin_angle))
    # 调整旋转矩阵的平移部分
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    # 执行旋转
    rotated_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))
    # 创建黑色背景
    background = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    # 将旋转后的图片放置在背景中
    background[:rotated_img.shape[0], :rotated_img.shape[1]] = rotated_img
    return background

# 示例调用
# img = cv2.imread('samples/bottle1.jpg')
# background = rotate_image_with_background(img)
# cv2.imshow('Original Image', img)
# cv2.imshow('Rotated Image with Background', background)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
