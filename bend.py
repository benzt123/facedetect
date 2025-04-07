import dlib, cv2
import numpy as np

# 加载检测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
img = cv2.imread("c.jpg")  # 替换为你的图片路径
# 检测特征点
faces = detector(img)
landmarks = predictor(img, faces[0])

# 定义眉毛移动向量（下压5像素）
brow_points = [18,19,20,21,22,23,24,25]  # 右眉特征点索引
src_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in brow_points])
dst_pts = src_pts + np.array([0,5])  # Y轴下移
cols, rows = img.shape[1], img.shape[0]
# 执行局部仿射变换
M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])  # 取前3点计算变换矩阵
result = cv2.warpAffine(img, M, (cols,rows))
cv2.imwrite("output.jpg", result)  # 保存结果