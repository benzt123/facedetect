import dlib
import cv2
import numpy as np
import math


class FaceAdjust():
    def __init__(self):

        self.predictor_path = "shape_predictor_68_face_landmarks.dat"
        #获取人脸框检测器
        self.detector = dlib.get_frontal_face_detector()
        #获取人脸关键点检测器
        self.predictor = dlib.shape_predictor(self.predictor_path)


    def landmark_dec_dlib_fun(self,img_src):
        img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
        #用于绘制人脸关键点的img
        img_tag = img_src.copy()

        land_marks = []
        rects = self.detector(img_gray, 0)
        
        for i in range(len(rects)):
            #把人脸关键点列表转为矩阵保存至列表中
            land_marks_node = np.matrix([[p.x, p.y] for p in self.predictor(img_gray, rects[i]).parts()])        
            land_marks.append(land_marks_node)
            #在图中标记出人脸关键点
            for j in land_marks_node:
                pt_pos = (j[0,0],j[0,1])
                cv2.circle(img_tag, pt_pos, 2, (0, 255, 0), -1)
            
        return land_marks


    '''
    localTranslationWarp:Interactive Image Warping 局部平移算法（移动圆内的像素）
    具体实现步骤
    2.1 对于圆形选区里的每一像素，取出其R,G,B各分量，存入3个变量（r, g, b）中（也即，三个变量分别存储选区内的原图像的R,G,B三个通道的数值）
    2.2 对于圆形选区里的每一个像素U
    2.3 根据图像局部平移变形，算出它变形后的位置坐标精确值X
    2.4 用插值方法，根据U的位置，和r, g, b中的数值，计算U所在位置处的R,G,B等分量，将R，G，B等分量合成新的像素，作为X处的像素值（bilinearInsert方法）
    '''


    def localTranslationWarp(self,srcImg, startX, startY, endX, endY, radius):
        ddradius = float(radius * radius)
        copyImg = np.zeros(srcImg.shape, np.uint8)
        copyImg = srcImg.copy()
        # 计算公式中的|m-c|^2
        ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
        W, H, C = srcImg.shape
        w1=max(startX-radius,0)
        w2=min(startX+radius,W)
        h1=max(startY-radius,0)
        h2=min(startY+radius,H)
        for i in range(int(w1),int(w2)):
            for j in range(int(h1),int(h2)):
                # 计算该点是否在形变圆的范围之内
                # 优化，第一步，直接判断是会在（startX,startY)的矩阵框中
                if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
                    continue

                distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)

                if (distance < ddradius):
                    # 计算出（i,j）坐标的原坐标
                    # 计算公式中右边平方号里的部分
                    ratio = (ddradius - distance) / (ddradius - distance + ddmc)
                    ratio = ratio * ratio

                    # 映射原位置(向后变形，j后+变-即为向前变形)
                    UX = i + ratio * (endX - startX)
                    UY = j + ratio * (endY - startY)

                    # 根据双线性插值法得到UX，UY的值
                    value = self.bilinearInsert(srcImg, UX, UY)
                    # 改变当前 i ，j的值
                    copyImg[j, i] = value

        return copyImg

    """
    bilinearInsert:双线性插值法（变换像素）
    """
    def bilinearInsert(self,src, ux, uy):
        w, h, c = src.shape
        if c == 3:
            x1 = int(ux)
            x2 = x1 + 1
            y1 = int(uy)
            y2 = y1 + 1

            part1 = src[y1, x1].astype(np.float64) * (float(x2) - ux) * (float(y2) - uy)
            part2 = src[y1, x2].astype(np.float64) * (ux - float(x1)) * (float(y2) - uy)
            part3 = src[y2, x1].astype(np.float64) * (float(x2) - ux) * (uy - float(y1))
            part4 = src[y2, x2].astype(np.float64) * (ux - float(x1)) * (uy - float(y1))

            insertValue = part1 + part2 + part3 + part4

            return insertValue.astype(np.int8)


    def face_adjust_auto(self,src):
        #1.获取人脸关键点
        landmarks = self.landmark_dec_dlib_fun(src)

        # 如果未检测到人脸关键点，就不进行脸部微调
        if len(landmarks) == 0:
            return

        
        thin_image = src
        #2.对鼻子为中心部分进行局部放大算法
        for i in range(len(landmarks)):
            landmarks_node = landmarks[i]
            start_landmark1 = landmarks_node[27]
            end_landmark1 = landmarks_node[31]
            start_landmark2 = landmarks_node[27]
            end_landmark2 = landmarks_node[35]
            start_landmark3 = landmarks_node[31]
            end_landmark3 = landmarks_node[51]
            start_landmark4 = landmarks_node[35]
            end_landmark4 = landmarks_node[51]
            r = math.sqrt((start_landmark1[0, 0] - end_landmark1[0, 0]) * (start_landmark1[0, 0] - end_landmark1[0, 0]) +
                (start_landmark1[0, 1] - end_landmark1[0, 1]) * (start_landmark1[0, 1] - end_landmark1[0, 1]))
            thin_image = self.localTranslationWarp(thin_image, start_landmark1[0, 0], start_landmark1[0, 1], end_landmark1[0,0], end_landmark1[0, 1], r)
            thin_image = self.localTranslationWarp(thin_image, start_landmark2[0, 0], start_landmark2[0, 1], end_landmark2[0,0], end_landmark2[0, 1], r)
            r = math.sqrt((start_landmark3[0, 0] - end_landmark3[0, 0]) * (start_landmark3[0, 0] - end_landmark3[0, 0]) +
                (start_landmark3[0, 1] - end_landmark3[0, 1]) * (start_landmark3[0, 1] - end_landmark3[0, 1]))
            thin_image = self.localTranslationWarp(thin_image, start_landmark3[0, 0], start_landmark3[0, 1], end_landmark3[0,0], end_landmark3[0, 1], r)
            thin_image = self.localTranslationWarp(thin_image, start_landmark4[0, 0], start_landmark4[0, 1], end_landmark4[0,0], end_landmark4[0, 1], r)
            # 显示
            # cv2.imshow('thin', thin_image)
        cv2.imwrite(r'mei.jpg', thin_image)


if __name__ == '__main__':
    src = cv2.imread(r'1-4k.JPG')
    face_adjust_control = FaceAdjust()
    face_adjust_control.face_adjust_auto(src)

