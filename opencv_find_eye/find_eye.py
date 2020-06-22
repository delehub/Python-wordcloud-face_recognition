'''
作者：树莓
原理：通过面部识别和眼部模型识别找出面部和眼部
      再把两个眼睛分别截取出来（剔除眉毛干扰）
      做二值化处理，寻找轮廓，对眼珠进行圆形拟合
      效果其实一般般
'''
#需要安装上面的两个库
import numpy as np
import cv2


def nothing(args):
    pass

#两个分类器目录一定要选对，就是文件夹里面的两个文件
# 人脸识别分类器 //新建立一个文件夹opencv_find_eye 在里面放入两个xml文件，把下面的路径改成你文件夹保存的路径
faceCascade = cv2.CascadeClassifier(r"E:\opencv face\opencv_find_eye\haarcascade_frontalface_default.xml")

# 识别眼睛的分类器
eyeCascade = cv2.CascadeClassifier(r'E:\opencv face\opencv_find_eye\haarcascade_eye.xml')

# 开启摄像头
cap = cv2.VideoCapture(0)
ok=True
cv2.namedWindow('eye1')
cv2.namedWindow('eye2')
#这两个滑块用来调整阈值
cv2.createTrackbar("th","eye1",20,255,nothing)
cv2.createTrackbar("th","eye2",20,255,nothing)
while ok:
    # 读取摄像头中的图像，ok为是否读取成功的判断参数
    ok, img = cap.read()
    img2 =img
    # 转换成灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(32, 32)
    )

    # 在检测人脸的基础上检测眼睛
    for (x, y, w, h) in faces:
        fac_gray = gray[y: (y+h), x: (x+w)]
        result = []
        eyes = eyeCascade.detectMultiScale(fac_gray, 1.3, 8,cv2.CASCADE_SCALE_IMAGE,(40,40),(80,80))

        # 眼睛坐标的换算，将相对位置换成绝对位置
        for (ex, ey, ew, eh) in eyes:
            #print(eyes[0])
            result.append((x+ex, y+ey, ew, eh))

    # 画矩形
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    try :
        for (ex, ey, ew, eh) in result:
            cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)#框出眼睛
            if result[0][0] >result[1][0]:     #判断是左眼还是右眼
                img_eye_l = cv2.resize(img2[result[0][1]:result[0][1]+result[0][3], result[0][0]:result[0][0]+result[0][2]], (300, 300))
                img_eye_r = cv2.resize(img2[result[1][1]:result[1][1]+result[1][3], result[1][0]:result[1][0]+result[1][2]], (300, 300))
            if result[0][0] <result[1][0]:
                img_eye_l = cv2.resize(img2[result[1][1]:result[1][1]+result[1][3], result[1][0]:result[1][0]+result[1][2]], (300, 300))
                img_eye_r = cv2.resize(img2[result[0][1]:result[0][1]+result[0][3], result[0][0]:result[0][0]+result[0][2]], (300, 300))
            
            img_eye_l = img_eye_l[60:240,10:290] #复制眼睛照片
            img_eye_r = img_eye_r[60:240,10:290]

            img_eye_l_gray = cv2.cvtColor(img_eye_l,cv2.COLOR_RGB2GRAY)  #转换为灰度图
            img_eye_r_gray = cv2.cvtColor(img_eye_r,cv2.COLOR_RGB2GRAY)

            X1 = cv2.getTrackbarPos("th","eye1") #获取滑块的值
            X2 = cv2.getTrackbarPos("th","eye2")

            ret1,adaptive_l = cv2.threshold(img_eye_l_gray,X1,255,cv2.THRESH_BINARY_INV) #普通二值化处理
            ret2,adaptive_r = cv2.threshold(img_eye_r_gray,X2,255,cv2.THRESH_BINARY_INV)

            #adaptive_l  = cv2.adaptiveThreshold(img_eye_l_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,81,2) #自适应二值化 效果并不好
            #adaptive_r = cv2.adaptiveThreshold(img_eye_r_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,81,2)


            mask_l = cv2.erode(adaptive_l, None, iterations=2) #收缩
            mask_r = cv2.erode(adaptive_r, None, iterations=2)

            mask_l = cv2.dilate(mask_l, None, iterations=2) #膨胀     收缩膨胀主要是为了过滤噪点
            mask_r = cv2.dilate(mask_r, None, iterations=2)

            cnts_l = cv2.findContours(mask_l.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  #寻找轮廓 简单轮廓

            cnts_r = cv2.findContours(mask_r.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            if len(cnts_l) >0: #圆形拟合
                c = max(cnts_l, key=cv2.contourArea)

                ((x, y), radius) = cv2.minEnclosingCircle(c)

                M = cv2.moments(c)

                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                if radius > 3:

                    cv2.circle(img_eye_l, (int(x), int(y)), int(radius), (255, 0, 0), 2)

                    cv2.circle(img_eye_l, center, 5, (0, 0, 255), -1)
            if len(cnts_r) >0: #圆形拟合
                c = max(cnts_r, key=cv2.contourArea)

                ((x, y), radius) = cv2.minEnclosingCircle(c)

                M = cv2.moments(c)

                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                if radius > 3:
                    

                    cv2.circle(img_eye_r, (int(x), int(y)), int(radius), (255, 0, 0), 2)

                    cv2.circle(img_eye_r, center, 5, (0, 0, 255), -1)

            

            cv2.imshow("eye1",img_eye_l)
            cv2.imshow("eye2",img_eye_r)

            #cv2.imshow("eye_l",img_eye_l_gray)
            #cv2.imshow("eye_r",img_eye_r_gray)

            #cv2.imshow("eye_l",adaptive_l)
            #cv2.imshow("eye_r",adaptive_r)
            
    except:
        pass

    cv2.imshow('video', img)
    

    k = cv2.waitKey(1)
    if k == 27:    # press 'ESC' to quit
        break
 
cap.release()
cv2.destroyAllWindows()
