import  cv2
from Handwritten import img_to_str


if __name__ == '__main__':
    # 创建一个窗口 1表示不能改变窗口大小
    cv2.namedWindow("camera",1)
    # 开启ip摄像头
    video = 'http://admin:admin@192.168.137.53:8081/video'
    # 开启摄像头
    capture = cv2.VideoCapture(video)
    # 按键处理
    while True:
        success,img = capture.read()
        cv2.imshow("camera",img)

        # 处理
        key = cv2.waitKey(10)
        if key == 27:

            print("esc break")
            break
        if key ==32:
            filename = "filename.png"
            cv2.imwrite(filename,img)
            s = img_to_str(filename)
            print(s)
    # 释放摄像头
    capture.release()
    #关闭窗口
    cv2.destroyWindow('camera')




