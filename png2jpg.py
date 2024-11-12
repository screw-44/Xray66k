import cv2
import numpy as np

# 读取带有透明背景的PNG图像
png_image = cv2.imread('samples/gun1.png', cv2.IMREAD_UNCHANGED)

# 分离图像的RGBA通道
b_channel, g_channel, r_channel, alpha_channel = cv2.split(png_image)

# 创建一个白色背景
white_background = np.ones_like(b_channel, dtype=np.uint8) * 255

# 使用alpha通道将原图像与白色背景混合
alpha_factor = alpha_channel.astype(np.float32) / 255.0
foreground = cv2.merge((b_channel, g_channel, r_channel))
background = cv2.merge((white_background, white_background, white_background))
result_image = cv2.convertScaleAbs(foreground * alpha_factor[:, :, None] + background * (1 - alpha_factor[:, :, None]))

# 保存结果图像为JPG格式
cv2.imwrite('samples/gun1.jpg', result_image)
