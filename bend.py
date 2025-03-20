import cv2
import numpy as np

def smooth_bend_transform(image, regions):
    """
    对图像的多个长方形区域进行平滑倾斜弯曲变换
    :param image: 输入图像
    :param regions: 长方形区域的参数列表，每个区域包含以下参数：
        - x_start, x_end: 水平范围
        - y_start, y_end: 垂直范围
        - amplitude: 弯曲幅度
        - frequency: 弯曲频率
        - phase_start: 正弦函数的初始相位
        - phase_step: 正弦函数的步长
    :return: 变换后的图像
    """
    result = image.copy()
    height, width = image.shape[:2]

    # 遍历每个长方形区域
    for region in regions:
        x_start, x_end = region["x_start"], region["x_end"]
        y_start, y_end = region["y_start"], region["y_end"]
        amplitude = region["amplitude"]
        frequency = region["frequency"]
        phase_start = region["phase_start"]
        phase_step = region["phase_step"]

        # 遍历长方形区域
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                # 计算正弦函数的相位
                phase = phase_start + (x - x_start) * phase_step
                # 计算垂直方向的偏移量
                offset = int(amplitude * np.sin(phase))

                # 计算新坐标
                y_new = y + offset

                # 边界检查
                if 0 <= y_new < height:
                    # 线性插值
                    if y_new + 1 < height:
                        # 计算权重
                        a = y_new - int(y_new)
                        # 线性插值
                        value = (1 - a) * image[int(y_new), x] + a * image[int(y_new) + 1, x]
                    else:
                        value = image[int(y_new), x]

                    # 更新结果图像
                    result[y, x] = value

    return result

# 创建网格图像
image = np.zeros((300, 300, 3), dtype=np.uint8)
for i in range(0, 300, 20):
    cv2.line(image, (i, 0), (i, 300), (255, 255, 255), 1)
    cv2.line(image, (0, i), (300, i), (255, 255, 255), 1)

# 定义多个长方形区域的参数
regions = [
    {
        "x_start": 50, "x_end": 150,  # 水平范围
        "y_start": 50, "y_end": 150,  # 垂直范围
        "amplitude": 100,              # 弯曲幅度
        "frequency": 1,               # 弯曲频率
        "phase_start": np.pi,         # 正弦函数的初始相位（第三个四分之一周期）
        "phase_step": np.pi /4 / (150 - 50)  # 正弦函数的步长
    },
    {
        "x_start": 200, "x_end": 300,  # 水平范围
        "y_start": 100, "y_end": 200,  # 垂直范围
        "amplitude": 100,               # 弯曲幅度
        "frequency": 1,                # 弯曲频率
        "phase_start": 3 * np.pi / 2,  # 正弦函数的初始相位（第四个四分之一周期）
        "phase_step": np.pi /4 /(300 - 200)  # 正弦函数的步长
    }
]

# 应用平滑倾斜弯曲变换
result = smooth_bend_transform(image, regions)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Smoothed Bend Transform', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
