import cv2
import numpy as np
import dlib

# -------------------- 初始化部分 --------------------
# 下载预训练模型并修改路径
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
FACE_DETECTOR = dlib.get_frontal_face_detector()
LANDMARK_PREDICTOR = dlib.shape_predictor(PREDICTOR_PATH)

# -------------------- 核心函数定义 --------------------
def get_roi_from_landmarks(img, landmarks, expand_ratio=0.2):
    """获取扩展后的矩形ROI区域"""
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    w = x_max - x_min
    h = y_max - y_min
    expand_w = int(w * expand_ratio)
    expand_h = int(h * expand_ratio)
    
    x_start = max(0, x_min - expand_w)
    y_start = max(0, y_min - expand_h)
    x_end = min(img.shape[1], x_max + expand_w)
    y_end = min(img.shape[0], y_max + expand_h)
    
    roi = img[y_start:y_end, x_start:x_end]
    return roi, (x_start, y_start, x_end-x_start, y_end-y_start)

def create_deformation_mesh(roi_shape, grid_size=(5,5)):
    """创建变形网格"""
    rows, cols = grid_size
    h, w = roi_shape[:2]
    return np.array([[[x*w/(cols-1), y*h/(rows-1)] 
                     for x in range(cols)] 
                     for y in range(rows)], dtype=np.float32)

def apply_frown_deformation(src_mesh, intensity=2.0):
    """应用皱眉变形"""
    dst_mesh = src_mesh.copy()
    
    # 内侧点向下位移
    dst_mesh[2:4, 0:3, 1] += 10 * intensity  # y坐标增加
    
    # 中间区域轻微下压
    dst_mesh[1:3, 3:5, 1] += 5 * intensity
    
    return dst_mesh

def mesh_warp(img, src_mesh, dst_mesh):
    """执行网格变形"""
    h, w = img.shape[:2]
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)
    
    # 生成亚像素映射
    for y in range(src_mesh.shape[0]-1):
        for x in range(src_mesh.shape[1]-1):
            src_pts = src_mesh[y:y+2, x:x+2].reshape(4,2).astype(np.float32)
            dst_pts = dst_mesh[y:y+2, x:x+2].reshape(4,2).astype(np.float32)
            
            # 计算局部变换矩阵
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # 计算网格区域
            x_start = int(src_mesh[y,x,0])
            y_start = int(src_mesh[y,x,1])
            x_end = int(src_mesh[y,x+1,0])
            y_end = int(src_mesh[y+1,x,1])
            
            # 生成映射
            for gy in range(y_start, y_end):
                for gx in range(x_start, x_end):
                    if 0 <= gx < w and 0 <= gy < h:
                        src_point = np.array([[[gx, gy]]], dtype=np.float32)
                        dst_point = cv2.perspectiveTransform(src_point, M)
                        map_x[gy, gx] = dst_point[0,0,0]
                        map_y[gy, gx] = dst_point[0,0,1]
    
    # 执行重映射
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

# -------------------- 主处理流程 --------------------
def main(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        return
    
    # 人脸检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_DETECTOR(gray)
    if len(faces) == 0:
        print("未检测到人脸")
        return
    
    # 关键点检测
    landmarks = LANDMARK_PREDICTOR(gray, faces[0])
    left_brow = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17,22)]
    
    # 截取ROI
    roi, (x,y,w,h) = get_roi_from_landmarks(img, left_brow, expand_ratio=5.0)
    if roi.size == 0:
        print("ROI截取失败")
        return
    
    # 创建变形网格
    src_mesh = create_deformation_mesh(roi.shape, grid_size=(5,5))
    dst_mesh = apply_frown_deformation(src_mesh, intensity=2.0)
    
    # 执行变形
    warped_roi = mesh_warp(roi, src_mesh, dst_mesh)
    
    # 创建融合掩膜
    mask = 255 * np.ones(roi.shape, dtype=np.uint8)
    
    # 泊松融合
    center = (x + w//2, y + h//2)  # 计算融合中心
    result = cv2.seamlessClone(warped_roi, img, mask, center, cv2.NORMAL_CLONE)
    
    # 显示结果
    cv2.imwrite(r'Frown.jpg', result)

# -------------------- 执行示例 --------------------
if __name__ == "__main__":
    # 替换为你的测试图像路径
    main("c.jpg")
