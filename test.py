import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = 'Data/8B0740EPAG531D1_1683558061000.jpg'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# 对Y通道应用直方图均衡化
yuv_image[:,:,0] = cv2.equalizeHist(yuv_image[:,:,0])

# 将图像转回BGR颜色空间
enhanced_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

# 对水平方向应用Sobel算子
sobelx = cv2.Sobel(enhanced_image, cv2.CV_64F, 1, 0, ksize=3)

# 对垂直方向应用Sobel算子
sobely = cv2.Sobel(enhanced_image, cv2.CV_64F, 0, 1, ksize=3)

# 计算梯度幅值
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)

# 转换为8位无符号整数类型
gradient_magnitude = np.uint8(gradient_magnitude)

blurred = cv2.GaussianBlur(gradient_magnitude, (5, 5), 0)
edges = cv2.Canny(blurred, 30, 80)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image', enhanced_image)
cv2.imshow('gradient Image', gradient_magnitude)
cv2.imshow('Contours', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()