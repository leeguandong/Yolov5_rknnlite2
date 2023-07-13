'''
@Time    : 2023/3/3 10:05
@Author  : leeguandon@gmail.com
'''
import numpy as np
import cv2

# 读取图像
img = cv2.imread("data/test.png")
# 坐标点points
pts = np.array([[10, 10], [15, 0], [35, 8], [100, 20], [300, 45], [280, 100], [350, 230], [30, 200]])
pts = np.array([pts])
# 和原始图像一样大小的0矩阵，作为mask
mask = np.zeros(img.shape[:2], np.uint8)
# 在mask上将多边形区域填充为白色
cv2.polylines(mask, pts, 1, 255)  # 描绘边缘
cv2.fillPoly(mask, pts, 255)  # 填充
# 逐位与，得到裁剪后图像，此时是黑色背景
dst = cv2.bitwise_and(img, img, mask=mask)
# 添加白色背景
bg = np.ones_like(img, np.uint8) * 255
cv2.bitwise_not(bg, bg, mask=mask)  # bg的多边形区域为0，背景区域为255
dst_white = bg + dst

cv2.imwrite("mask.jpg", mask)
cv2.imwrite("dst.jpg", dst)
cv2.imwrite("dst_white.jpg", dst_white)
