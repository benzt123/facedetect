import cv2
from math import sin,cos,pi
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
def landmark_dec_fun(img_path):
    img = cv2.imread(img_path)
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
    cv2.imwrite(r'tagc.jpg', img)
    return s
cnt=0

def lens_distortion(x, y, cx, cy, k, a, b):
    global cnt
    # 转换为椭圆归一化坐标
    u = (x*1. - cx*1.) / (a*1.)
    v = (y*1.- cy*1.) / (b*1.)
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
    x_prime = u_prime * a + cx
    y_prime = v_prime * b + cy
    if cnt<=10:
        cnt+=1
        print(x,y,x_prime,y_prime)
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
                if ((x - center[0])**2 / axes[0]**2 + (y - center[1])**2 / axes[1]**2) <= 1:
                    # 应用透镜变换
                    x_new, y_new = lens_distortion(x, y, center[0], center[1], k,axes[0],axes[1])
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
    ellipses = []
    for i in range(len(faces)):
        ellipse = {"center": (100, 100), "axes": (50, 30), "angle": 0, "k": 0.01}
        s=faces[i]
        x1=s[t1][0];y1=s[t1][1]
        x2=s[t2][0];y2=s[t2][1]
        x3=s[t3][0];y3=s[t3][1]
        cx=int((x1+x2+x3)/3)
        cy=int((y1+y2+y3)/3)
        ellipse = {"center": (cx, cy), "axes": (abs(y1-y3)//2,abs(x2-x1)), "angle": 0, "k": 1.5}
        print(ellipse)
        ellipses.append(ellipse)
    result = apply_lens_effect(img, ellipses)
    return result
path="c.JPG"
faces=landmark_dec_fun(path)
result=change(path,faces)
cv2.imwrite(r'1-4t-1-1.4.jpg', result)