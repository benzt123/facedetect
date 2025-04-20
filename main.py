# 核心FastAPI实现
from fastapi import FastAPI, UploadFile, File
import base64
from face import FaceTransformer
import cv2
app = FastAPI()

@app.post("/transform/{transform_type}")
async def transform_face(
    transform_type: int, 
    image: UploadFile = File(...),
):
    print(transform_type)
    # 保存上传文件
    input_path = "temp/input.jpg"
    with open(input_path, "wb") as f:
        f.write(await image.read())
    
    # 初始化变形器
    ft = FaceTransformer()
    ft._get_landmarks(input_path)
    # 根据类型选择处理方法
    result1 = ft.change1()
    overlay_path = "zdhx2.jpg"
    result2 = ft.change2(overlay_path)
    result3 = ft.change3()
    _, buffer1 = cv2.imencode('.jpg', result1)
    img1 = base64.b64encode(buffer1).decode()

    # 处理类型2，需要overlay
    print(2)
    _, buffer2 = cv2.imencode('.jpg', result2)
    img2 = base64.b64encode(buffer2).decode()
    print(3)
    _, buffer3 = cv2.imencode('.jpg', result3)
    img3 = base64.b64encode(buffer3).decode()

    # 返回三个结果
    return {
        "image1": img1,
        "image2": img2,
        "image3": img3
    }


