import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图片
img = cv2.imread('samples/bottle1.jpg')

# 将图像转换为HSV色彩空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义黑色背景的颜色范围
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

# 创建掩码，选择黑色区域
mask = cv2.inRange(hsv, lower_black, upper_black)

# 反转掩码，选择非黑色区域（瓶子）
mask_inv = cv2.bitwise_not(mask)

# 保留原图中的瓶子部分
result = cv2.bitwise_and(img, img, mask=mask_inv)

# 将黑色背景区域设置为白色（可选步骤）
background = np.full_like(img, [255, 255, 255])  # 创建一个白色背景
final_img = np.where(result == 0, background, result)  # 替换黑色部分为白色

# 显示结果
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('Mask (Black Background)')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
plt.title('Final Image (No Black Background)')

plt.show()
