import cv2
import numpy as np

def lens_distortion(x, y, cx, cy, k):
    # 径向畸变映射函数
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    x_new = x + (x - cx) * k * r
    y_new = y + (y - cy) * k * r
    return x_new, y_new

def linear_interpolation(image, x, y):
    # 线性插值
    x1, y1 = int(x), int(y)
    x2, y2 = x1 + 1, y1 + 1

    # 边界检查
    if x2 >= image.shape[1] or y2 >= image.shape[0]:
        return image[y1, x1]

    # 计算权重
    a = x - x1
    b = y - y1

    # 线性插值
    value = (1 - a) * (1 - b) * image[y1, x1] + \
            a * (1 - b) * image[y1, x2] + \
            (1 - a) * b * image[y2, x1] + \
            a * b * image[y2, x2]
    return value

def apply_lens_effect(image, center, axes, angle, k):
    # 创建结果图像
    result = np.zeros_like(image)

    # 遍历椭圆区域
    for y in range(center[1] - axes[1], center[1] + axes[1]):
        for x in range(center[0] - axes[0], center[0] + axes[0]):
            # 检查是否在椭圆内
            if ((x - center[0])**2 / axes[0]**2 + (y - center[1])**2 / axes[1]**2) <= 1:
                # 应用透镜变换
                x_new, y_new = lens_distortion(x, y, center[0], center[1], k)

                # 线性插值
                if 0 <= x_new < image.shape[1] and 0 <= y_new < image.shape[0]:
                    result[y, x] = linear_interpolation(image, x_new, y_new)

    return result

# 创建网格图像
image = np.zeros((300, 300, 3), dtype=np.uint8)
for i in range(0, 300, 20):
    cv2.line(image, (i, 0), (i, 300), (255, 255, 255), 1)
    cv2.line(image, (0, i), (300, i), (255, 255, 255), 1)

# 定义椭圆参数
center = (image.shape[1] // 2, image.shape[0] // 2)
axes = (100, 50)  # 长轴和短轴
angle = 0  # 旋转角度
k = 0.02  # 畸变系数

# 应用透镜效果
result = apply_lens_effect(image, center, axes, angle, k)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Lens Effect', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
