import cv2
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


def exaggerate(img1,img,center,radius):
    r=radius
    k=0.8
    pk=0.5
    w1,w2,h1,h2=max(center[0]-r,0),min(center[0]+r,img.shape[0]),max(center[1]-r,0),min(center[1]+r,img.shape[1])
    for i in range(int(w1),int(w2)):
        for j in range(int(h1),int(h2)):
            #局部放大
            d_square=(i-center[0])**2+(j-center[1])**2
            if d_square>r**2 :
                continue
            if d_square<=(k*r)**2 :
                ux1=int(float(center[0])+float(i-center[0])/k*pk)
                if i-center[0]!=0:
                    ux2=ux1+(i-center[0])/abs(i-center[0])
                else:
                    ux2=ux1
                uy1=int(float(center[1])+float(j-center[1])/k*pk)
                if j-center[1]!=0:
                    uy2=uy1+(j-center[1])/abs(j-center[1])
                else:
                    uy2=uy1
                # bilinear interpolation
                dx=float(ux1-center[0])/k
                img1[j,i]=img[uy1,ux1]
                #img1[i,j]=img[ux1,uy1].astype(np.float64)+(img[ux2,uy2].astype(np.float64)-img[ux1,uy1].astype(np.float64))*(dx - float(ux1 - center[0]))*pk/k
            else:
                #局部缩小
                dx=float(i-center[0])/(float(d_square)**0.5)*(float(r))
                dy=float(j-center[1])/(float(d_square)**0.5)*(float(r))
                ux=int(float(center[0])+(float(i-center[0])-dx*k)/(1-k)*pk+dx*pk)
                uy=int(float(center[1])+(float(j-center[1])-dy*k)/(1-k)*pk+dy*pk)
                img1[j,i]=img[uy,ux]
def change(img_path,faces):
    img=cv2.imread(img_path)
    img1=img.copy()
    for i in range(len(faces)):
        s=faces[i]
        x1=s[17][0];y1=s[17][1]
        x2=s[26][0];y2=s[26][1]
        x3=s[62][0];y3=s[62][1]
        cx=int((x1+x2+x3)/3)
        cy=int((y1+y2+y3)/3)
        r=int(((x1-cx)**2+(y1-cy)**2)**0.5) # 半径
        exaggerate(img1,img,[cx,cy],r*0.8)
    cv2.imwrite(r'c.jpg', img1)
path="1-4k.JPG"
faces=landmark_dec_fun(path)
change(path,faces)