import cv2

# 读取图片
img = cv2.imread('zdhx.jpg')
print(img.shape)
x,y=img.shape[0],img.shape[1]
for i in range(x):
    print(len(img[i]))
