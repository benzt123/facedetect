import cv2
from math import sin,cos,pi,asin,acos
"""
This script performs face detection and landmark identification on an input image using OpenCV and dlib libraries.
Functions:
    - cv2.imread: Reads an image from a file.
    - cv2.cvtColor: Converts an image from one color space to another.
    - dlib.get_frontal_face_detector: Returns a face detector object.
    - dlib.shape_predictor: Loads a pre-trained shape predictor model.
    - detector: Detects faces in the input image.
    - predictor: Predicts facial landmarks for detected faces.
    - cv2.circle: Draws circles on the image at specified points.
    - cv2.namedWindow: Creates a window that can be resized.
    - cv2.imshow: Displays an image in the specified window.
    - cv2.waitKey: Waits for a key event.
    - cv2.destroyAllWindows: Destroys all the windows created.
Variables:
    - img_path (str): Path to the input image file.
    - img (numpy.ndarray): The input image.
    - size (tuple): Dimensions of the input image.
    - w (int): Width of the input image.
    - h (int): Height of the input image.
    - gray (numpy.ndarray): Grayscale version of the input image.
    - detector (dlib.fhog_object_detector): Face detector object.
    - predictor (dlib.shape_predictor): Shape predictor object.
    - faces (dlib.rectangles): Detected faces in the image.
    - shape (dlib.full_object_detection): Facial landmarks for a detected face.
    - s1 (list): List of facial landmark coordinates.
    - pt_pos (tuple): Coordinates of a facial landmark point.
"""
import dlib
import numpy as np
# 读取图片

class FaceAdjust():
    def __init__(self):
        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)
def landmark_dec_fun(img):
    size = img.shape
    w = size[1]
    h = size[0]
    # 转换为灰阶图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 正向人脸检测器
    detector = dlib.get_frontal_face_detector()
    # 使用训练完成的68个特征点模型
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    # 使用检测器来检测图像中的人脸
    faces = detector(gray, 1)
    s=[]
    for i, face in enumerate(faces):
        # 获取人脸特征点
        shape = predictor(img, face)
        s1 = []
        # 遍历所有点
        for pt in shape.parts():
            # 绘制特征点
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 5, (0,0,138), -1)
            s1.append([pt.x,pt.y])
        cx=int((s1[19][0]+s1[24][0]+s1[66][0])/3)
        cy=int((s1[19][1]+s1[24][1]+s1[66][1])/3)
        cv2.circle(img, (cx,cy), 5, (0,0,138), -1)
        s.append(s1.copy())
    return s
cnt=0

def lens_distortion(point, k,ellipse):
    global cnt
    # 转换为椭圆归一化坐标
    x, y = point
    cx, cy = ellipse["center"]
    a, b = ellipse["axes"]
    angle = ellipse["angle"]
    x -= cx
    y -= cy
    dx, dy = x*1. * cos(angle) + y*1. * sin(angle), -x*1. * sin(angle) + y*1. * cos(angle)
    # 计算径向距离
    u = dx/ (a*1.)
    v = dy/ (b*1.)
    r = np.sqrt(u**2 + v**2)
    
    # 应用径向变形函数（凸透镜）
    r_prime=sin(pi*r/2)/(1+k*cos(pi*r/2))
    if (r>0):
        u_prime = u * r_prime / r
        v_prime = v * r_prime / r
    else:
        u_prime = u
        v_prime = v
    # 逆映射回原椭圆坐标系
    dx_prime = u_prime * (a*1.)
    dy_prime = v_prime * (b*1.)
    x_prime = dx_prime * cos(angle) - dy_prime * sin(angle) + cx
    y_prime = dx_prime * sin(angle) + dy_prime * cos(angle) + cy

    return x_prime, y_prime

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

def apply_lens_effect(image, ellipses):
    # 创建结果图像
    result = image.copy()

    # 遍历每个椭圆区域
    for ellipse in ellipses:
        center = ellipse["center"]
        axes = ellipse["axes"]
        angle = ellipse["angle"]
        k = ellipse["k"]

        # 遍历椭圆区域
        flag=0
        for y in range(center[1] - axes[1], center[1] + axes[1]):
            for x in range(center[0] - axes[0], center[0] + axes[0]):
                # 检查是否在椭圆内
                xx,yy=0.,0.
                xx,yy=x-center[0],y-center[1]
                if (xx*cos(angle)+yy*sin(angle))**2 / axes[0]**2 + (xx*sin(angle)-yy*cos(angle))**2 / axes[1]**2 <= 1+1e-6:
                    # 应用透镜变换
                    x_new, y_new = lens_distortion((x, y), k,ellipse)
                    if x!=x_new or y!=y_new:
                        flag=1
                    # 线性插值
                    if 0 <= x_new < image.shape[1] and 0 <= y_new < image.shape[0]:
                        result[y, x] = linear_interpolation(image, x_new, y_new)
        if flag==0:
            print("no change")
    return result
def change(img_path,faces):
    img=cv2.imread(img_path)
    t1=19
    t2=25
    t3=66
    tc=27
    tc1=51
    ellipses = []
    for i in range(len(faces)):
        ellipse = {"center": (100, 100), "axes": (50, 30), "angle": 0, "k": 0.01}
        s=faces[i]
        x1=s[t1][0];y1=s[t1][1]
        x2=s[t2][0];y2=s[t2][1]
        x3=s[t3][0];y3=s[t3][1]
        cx=int((x1+x2+x3)/3)
        cy=int((y1+y2+y3)/3)

        dx,dy=s[tc][0]-s[tc1][0],s[tc][1]-s[tc1][1]
        dis=np.sqrt(dx**2+dy**2)
        ang=asin(dx/dis)
        ellipse = {"center": (cx, cy), "axes": (abs(x2-x1)//2,abs(y1-y3)), "angle": ang, "k": 1.5}
        print(ellipse)
        ellipses.append(ellipse)
    result = apply_lens_effect(img, ellipses)
    return result

def overlay_image(background, overlay, x, y):
    """
    将overlay图像叠加到background的指定位置
    :param background: 背景图像
    :param overlay: 前景图像
    """
    # 获取前景图像的尺寸
    h, w = overlay.shape[:2]
    na = [[255,255,255],
          [255,255,255],
          [255,255,255]]
    for i in range(h):
        for j in range(w):
            if np.sum(np.square(overlay[i][j]-na))>1000:
                background[y+i][x+j]=overlay[i][j]
    
    return background
def change2(img_path,faces):
    img=cv2.imread(img_path)
    t1,t2=1,2
    t3,t4=15,14
    t5,t6=57,8
    tc=27
    tc1=51
    ellipses = []
    lefts = []
    for i in range(len(faces)):
        ellipse = {"center": (100, 100), "axes": (50, 30), "angle": 0, "k": 0.01}
        s=faces[i]
        x1=(s[t1][0]+s[t2][0])//2;y1=(s[t1][1]+s[t2][1])//2
        x2=(s[t3][0]+s[t4][0])//2;y2=(s[t3][1]+s[t4][1])//2
        x3=(s[t5][0]+s[t6][0])//2;y3=(s[t5][1]+s[t6][1])//2
        print(x1,y1,x2,y2,x3,y3)
        cx=int((x1+x2+x3)/3)
        cy=int((y1+y2+y3)/3)
        dx,dy=s[tc][0]-s[tc1][0],s[tc][1]-s[tc1][1]
        dis=np.sqrt(dx**2+dy**2)
        ang=asin(dx/dis)
        ellipse = {"center": (cx, cy), "axes": (abs(x2-x1)//2,abs(y1-y3)//3*2), "angle": ang, "k": 1.5}
        print(ellipse)
        ellipses.append(ellipse)
        zx1=s[17][0];zy1=s[17][1]-(s[57][1]-s[27][1]);
        a=s[26][0]-s[17][0];b=s[33][1]-s[27][1];
        lefts.append([(zx1,zy1),(a,b)])
    result = apply_lens_effect(img, ellipses)
    background = result
    overlay = cv2.imread('zdhx.jpg')
    for i in range(len(lefts)):
        x,y=lefts[i][0]
        a,b=lefts[i][1]
        overlay=cv2.resize(overlay,(a,b))
        background=overlay_image(background,overlay,x,y)
    return result


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
    import numpy as np

    def smooth_edge_mask(sh, decay_radius=50):
    # 生成边缘衰减掩模（0~1）
        mask = np.ones((height, width))
        for i in range(0, sh[0][1]-sh[0][0]):
            for j in range(0, sh[1][1]-sh[1][0]):
                # 计算到边界的距离
                d_left = j
                d_right = width - j
                d_top = i
                d_bottom = height - i
                d_min = min(d_left, d_right, d_top, d_bottom)
                # 高斯衰减：边缘decay_radius像素内逐渐衰减到0
                mask[i,j] = np.exp(-(max(0, decay_radius - d_min)**2) / (2*(decay_radius/3)**2))
        return mask

    result = image
    height, width = image.shape[:2]

    # 遍历每个长方形区域
    for region in regions:
        x_start, x_end = region["x_start"], region["x_end"]
        y_start, y_end = region["y_start"], region["y_end"]
        amplitude = region["amplitude"]
        frequency = region["frequency"]
        phase_start = region["phase_start"]
        phase_step = region["phase_step"]
        mask=smooth_edge_mask([(x_start, x_end),(y_start, y_end)], decay_radius=50)
        # 遍历长方形区域
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                # 计算正弦函数的相位
                phase = phase_start + (x - x_start) * phase_step
                
                # 修改扭曲函数，加入衰减
                offset = amplitude * np.sin(frequency * y + phase) #mask[y-y_start, x-x_start]
                # 计算新坐标
                y_new = y + offset
                if (y_new < y_start ):
                    y_new=y_start
                elif (y_new > y_end):
                    y_new=y_end
                value= linear_interpolation(image, x, y_new)
                result[y, x] = value

    return result
def change3(img_path,faces):
    img=cv2.imread(img_path)
    t1=19
    t2=21
    t3=22
    t4=24
    bends = []
    for i in range(len(faces)):
        s=faces[i]
        #左眉毛
        x1=s[t1][0];y1=s[t1][1]
        x2=s[t2][0];y2=s[t2][1]
        bend1 = {
        "y_start": y1-abs(x1-x2)//2, "y_end": y2+abs(x1-x2)//2,  # 水平范围
        "x_start": x1, "x_end": x2,  # 垂直范围
        "amplitude": 10,              # 弯曲幅度
        "frequency": 1,               # 弯曲频率
        "phase_start": np.pi,         # 正弦函数的初始相位（第三个四分之一周期）
        "phase_step": np.pi /4 / (abs(x2-x1))  # 正弦函数的步长
        }
        print(bend1)
        bends.append(bend1)

        #右眉毛
        x1=s[t3][0];y1=s[t3][1]
        x2=s[t4][0];y2=s[t4][1]
        
        bend2 = {
        "y_start": y1-abs(x1-x2)//2, "y_end": y2+abs(x1-x2)//2,  # 水平范围
        "x_start": x1, "x_end": x2,  # 垂直范围
        "amplitude": 10,              # 弯曲幅度
        "frequency": 1,               # 弯曲频率
        "phase_start": np.pi*3/2,         # 正弦函数的初始相位（第三个四分之一周期）
        "phase_step": np.pi /4 / ((y2+abs(x1-x2)//2)-(y1-abs(x1-x2)//2))  # 正弦函数的步长
        }
        print(bend2)
        bends.append(bend2)
    result = smooth_bend_transform(img, bends)
    return result

def main():
    path="c.JPG"
    img = cv2.imread(path)
    choose=3
    faces=landmark_dec_fun(img)
    if choose==1:
         result=change(path,faces)
    elif choose==2:
        result=change2(path,faces)
    elif choose==3:
        result=change3(path,faces)
    else:
        result=img
    cv2.imwrite(r'1-4t-sin-dx-3.jpg', result)
if __name__ == "__main__":
    main()
'''
'''