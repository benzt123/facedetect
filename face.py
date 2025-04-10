import cv2
import dlib
import numpy as np
from math import sin, cos, pi, asin, acos

class FaceTransformer:
    def __init__(self,predictor_path="shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.img = cv2.imread("./photos/6.JPG")
        self.landmarks = []
        img = self.img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        self.landmarks.clear()
        for face in faces:
            shape = self.predictor(img, face)
            self.landmarks.append([(pt.x, pt.y) for pt in shape.parts()])
    
    def _get_landmarks(self, img_path="./photos/6.JPG"):
        """通用特征点检测方法"""
        self.img = cv2.imread(img_path)
        img = self.img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        self.landmarks.clear()
        for face in faces:
            shape = self.predictor(img, face)
            self.landmarks.append([(pt.x, pt.y) for pt in shape.parts()])
        return img

    # -------------------- 变形方法1：鼻梁区域透镜变形 --------------------
    def change1(self):
        """鼻子区域变形 (基于19/24/66特征点)"""
        img= self.img.copy()
        ellipses = []
        
        # 鼻梁区域椭圆参数计算
        for face in self.landmarks:
            pt19, pt24, pt66 = face[19], face[24], face[66]
            cx = int((pt19[0] + pt24[0] + pt66[0])/3)
            cy = int((pt19[1] + pt24[1] + pt66[1])/3)
            
            # 椭圆角度计算（基于27/51号特征点）
            dx = face[27][0] - face[51][0]
            dy = face[27][1] - face[51][1]
            angle = asin(dx / np.hypot(dx, dy))
            
            ellipses.append({
                "center": (cx, cy),
                "axes": (abs(pt24[0]-pt19[0])//2, abs(pt19[1]-pt66[1])),
                "angle": angle,
                "k": 1.5
            })
        
        return self._apply_lens_effect(img, ellipses)

    # -------------------- 变形方法2：下半脸部变形+贴图 --------------------
    def change2(self,  overlay_path='zdhx2.jpg'):
        """下半脸变形与贴图 (基于1/2/15/14/57/8特征点)"""
        img = self.img.copy()
        ellipses = []
        overlay_positions = []
        t1,t2=1,2
        t3,t4=15,14
        t5,t6=57,8
        tc=27
        tc1=51
        for face in self.landmarks:
            # 下半脸椭圆参数
            s = face
            x1=(s[t1][0]+s[t2][0])//2;y1=(s[t1][1]+s[t2][1])//2
            x2=(s[t3][0]+s[t4][0])//2;y2=(s[t3][1]+s[t4][1])//2
            x3=(s[t5][0]+s[t6][0])//2;y3=(s[t5][1]+s[t6][1])//2
            #print(x1,y1,x2,y2,x3,y3)
            cx=int((x1+x2+x3)/3)
            cy=int((y1+y2+y3)/3)
            
            # 计算角度
            dx = s[27][0] - s[51][0]
            angle = asin(dx / np.hypot(dx, s[27][1]-s[51][1]))
            
            ellipses.append({"center": (cx, cy),
                             "axes": (abs(x2-x1)//2,abs(y1-y3)//3*2),
                             "angle": angle, 
                             "k": 1.5
            })
            
            # 贴图位置计算（17号特征点区域）
            x = s[17][0]
            y = s[17][1] - (s[57][1] - s[27][1])
            width = s[26][0] - s[17][0]
            height = int((s[33][1] - s[27][1]))
            overlay_positions.append((x, y, width, height))
        
        # 应用变形并叠加图像
        result = self._apply_lens_effect(img, ellipses)
        return self._apply_overlay(result, overlay_path, overlay_positions)

    # -------------------- 变形方法3：眉毛弯曲+嘴部变形 --------------------
    def change3(self):
        """组合变形：眉毛弯曲+嘴部变形"""
        img = self.img.copy()
        bend_regions = []
        mouth_ellipses = []
        t1=19
        t2=21
        t3=22
        t4=24
        for face in self.landmarks:
            # 眉毛弯曲参数（19/21和24/22号点）
            s=face
            #左眉毛
            x1=s[t1][0];y1=s[t1][1]
            x2=s[t2][0];y2=s[t2][1]
            bend1 = {
            "y_start": y1-abs(x1-x2), "y_end": y2+abs(x1-x2)//8,  # 水平范围
            "x_start": x1, "x_end": x2,  # 垂直范围
            "amplitude": abs(x1-x2)//2,              # 弯曲幅度
            "frequency": 1,               # 弯曲频率
            "phase_start": np.pi*3.0/2.0,         # 正弦函数的初始相位（第三个四分之一周期）
            "phase_step": np.pi/2 / (abs(x2-x1))  # 正弦函数的步长
            }

            #右眉毛
            x1=s[t4][0];y1=s[t4][1]
            x2=s[t3][0];y2=s[t3][1]
            
            bend2 = {
            "y_start": y1-abs(x1-x2), "y_end": y2+abs(x1-x2)//8,  # 水平范围
            "x_start": min(x1,x2), "x_end": max(x1,x2),  # 垂直范围
            "amplitude": abs(x1-x2)//2,              # 弯曲幅度
            "frequency": 1,               # 弯曲频率
            "phase_start": np.pi,         # 正弦函数的初始相位（第三个四分之一周期）
            "phase_step": np.pi /2/ (abs(x2-x1))  # 正弦函数的步长
            }
            bend_regions += [bend1, bend2]
            
            # 嘴部椭圆参数（48/61等特征点对）
            ce=[49,53,59,55]
            m=[(48,61), (53,54), (48,67), (65,54)]
            for i in range(4):
                x1=s[m[i][0]][0];y1=s[m[i][0]][1]
                x2=s[m[i][1]][0];y2=s[m[i][1]][1]
                cx,cy = (x1+x2)//2,(y1+y2)//2
                dx,dy = cx-s[ce[i]][0],cy-s[ce[i]][1]
                dis=np.sqrt(dx**2+dy**2)
                d = int(np.sqrt((x1-x2)**2+(y1-y2)**2))
                ang=asin(dx/dis)
                ellipse = {"center": (cx, cy), "axes": (d,d), "angle": ang, "k": 1.5}
                mouth_ellipses.append(ellipse)
        
        # 执行组合变形
        img = self._apply_bend_transform(img, bend_regions)
        return self._apply_lens_effect(img, mouth_ellipses)

    # -------------------- 通用变形算法 --------------------
    def _apply_lens_effect(self, img, ellipses):
        """透镜变形核心算法"""
        def lens_distortion(point, k, ellipse):
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
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + 1, y0 + 1
            if x0 < 0 or y0 < 0 or x1 >= image.shape[1] or y1 >= image.shape[0]:
                return image[y0, x0]
            a = x - x0
            b = y - y0
            return (1-a)*(1-b)*image[y0, x0] + a*(1-b)*image[y0, x1] + (1-a)*b*image[y1, x0] + a*b*image[y1, x1]
        result = img.copy()
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
                        if 0 <= x_new < img.shape[1] and 0 <= y_new < img.shape[0]:
                            result[y, x] = linear_interpolation(img, x_new, y_new)
            if flag==0:
                print("no change")
        return result

    def _apply_overlay(self, img, overlay_path, positions):
        """贴图叠加实现"""
        overlay = cv2.imread(overlay_path)
        """
        将overlay图像叠加到background的指定位置
        :param background: 背景图像
        :param overlay: 前景图像
        """
        # 获取前景图像的尺寸
        
        na = [[0,0,0],
            [0,0,0],
            [0,0,0]]
        for pos in positions:
            x= pos[0]
            y= pos[1]
            h= pos[2]
            w= pos[3]
            overlay1=cv2.resize(overlay,(h,w))
            for i in range(overlay1.shape[0]):
                for j in range(overlay1.shape[1]):
                    if np.sum(np.square(overlay1[i][j]-na))>1000:
                        img[y+i][x+j]=overlay1[i][j]
        
        return img

    def _apply_bend_transform(self, img, regions):
        """眉毛弯曲算法"""
        # 此处完整保留smooth_bend_transform算法
        def linear_interpolation(image, x, y):
            x0, y0 = int(x), int(y)
            x1, y1 = x0 + 1, y0 + 1
            if x0 < 0 or y0 < 0 or x1 >= image.shape[1] or y1 >= image.shape[0]:
                return image[y0, x0]
            a = x - x0
            b = y - y0
            return (1-a)*(1-b)*image[y0, x0] + a*(1-b)*image[y0, x1] + (1-a)*b*image[y1, x0] + a*b*image[y1, x1]
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

        result = img

        # 遍历每个长方形区域
        for region in regions:
            #print(region)
            x_start, x_end = region["x_start"], region["x_end"]
            y_start, y_end = region["y_start"], region["y_end"]
            amplitude = region["amplitude"]
            phase_start = region["phase_start"]
            phase_step = region["phase_step"]
            # 遍历长方形区域
            brow_center = (int((x_start + x_end) / 2), int((y_start + y_end) / 2))  
            strength = amplitude 
            for j in range(x_start, x_end):
                phase= phase_start + (j - x_start) * phase_step
                for i in range(y_start, y_end):
                    # 计算相对于眉毛中心的局部坐标
                    dx= (j - brow_center[0]) / (x_end - x_start)*2
                    dy= (i - brow_center[1]) / (y_end - y_start)*2
                    delta_y = 1.0*strength * (np.sin(phase)+1) * (1 - dx**2)*(1 - dy**2)
                    if (dy>0):
                        delta_y*=2
                    new_i = .0
                    new_i = i + delta_y
                    new_i = max(y_start, min(new_i, y_end))  # 限制在范围内
                    new_j = j  # 限制在范围内
                    value = linear_interpolation(img, j, new_i)
                    #if np.sum(np.square(img[i][j]-img[new_i]))>1000
                    result[i,j] = value
        return result

if __name__ == "__main__":
    img_path = "./photos/1.JPG"
    ft = FaceTransformer()
    ft._get_landmarks(img_path)
    result1 = ft.change1()
    cv2.imwrite("./photos/nose_lens.jpg", result1)
    
    result2 = ft.change2()
    cv2.imwrite("./photos/lower_face_overlay.jpg", result2)
    result3 = ft.change3()
    cv2.imwrite("./photos/brow_mouth_bend.jpg", result3)
