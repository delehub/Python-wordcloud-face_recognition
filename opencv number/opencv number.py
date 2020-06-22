import cv2
# from sklearn.externals import joblib
import joblib
import mahotas
import numpy as np
from sklearn.svm import LinearSVC
from skimage import feature


# 定义一个缩放函数
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    # 高度模式
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    # 宽度模式
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# 从excel加载数字，把特征和标注分开
def load_digits(datasetPath):
    data = np.genfromtxt(datasetPath, delimiter=",", dtype="uint8")
    target = data[:, 0]
    data = data[:, 1:].reshape(data.shape[0], 28, 28)
    return (data, target)


# 进行旋转变换
def deskew(image, width):
    (h, w) = image.shape[:2]
    moments = cv2.moments(image)
    skew = moments["mu11"] / moments["mu02"]
    M = np.float32([
        [1, skew, -0.5 * w * skew],
        [0, 1, 0]])
    image = cv2.warpAffine(image, M, (w, h),
                           flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    image = resize(image, width=width)
    return image


# 把数字缩放到图片中心
def center_extent(image, size):
    (eW, eH) = size

    # 如果宽度》高度
    if image.shape[1] > image.shape[0]:
        image = resize(image, width=eW)
    else:
        image = resize(image, height=eH)

    extent = np.zeros((eH, eW), dtype="uint8")
    offsetX = (eW - image.shape[1]) // 2
    offsetY = (eH - image.shape[0]) // 2
    extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] = image

    # 计算图片的质量中心
    (cY, cX) = np.round(mahotas.center_of_mass(extent)).astype("int32")
    (dX, dY) = ((size[0] // 2) - cX, (size[1] // 2) - cY)
    M = np.float32([[1, 0, dX], [0, 1, dY]])
    # 把质量中心移动到图片的中心
    extent = cv2.warpAffine(extent, M, size)

    # return the extent of the image
    return extent


class HOG:
    def __init__(self, orientations=9, pixelsPerCell=(8, 8),
                 cellsPerBlock=(3, 3), transform=False):
        self.orienations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.transform = transform

    def describe(self, image):
        hist = feature.hog(image, orientations=self.orienations,
                           pixels_per_cell=self.pixelsPerCell,
                           cells_per_block=self.cellsPerBlock,
                           transform_sqrt=self.transform)
        return hist


# 首先加载需要训练的数据
datasetPath = "E:/opencv face/opencv number/sample_submission.csv"
(digits, target) = load_digits(datasetPath)
data = []

# 初始化hog因子
hog = HOG(orientations=18, pixelsPerCell=(10, 10),
          cellsPerBlock=(1, 1), transform=True)

# 数据预处理
for image in digits:
    # 旋转和中心化
    image = deskew(image, 20)
    image = center_extent(image, (20, 20))
    # 使用hog算子描述图像特征
    hist = hog.describe(image)
    data.append(hist)

# 开始训练
model = LinearSVC(random_state=42)
model.fit(data, target)
myModel = "mysvm.cpickle"
joblib.dump(model, myModel)

# 加载训练好的模型
model = joblib.load(myModel)
hog = HOG(orientations=18, pixelsPerCell=(10, 10),
          cellsPerBlock=(1, 1), transform=True)
# 加载被分类的图片
imagePath = "E:/opencv face/opencv number/opencv_number.png"
image = cv2.imread(imagePath)
# 图片预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 30, 150)
# 根据轮廓对数字进行切分
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key=lambda x: x[1])
for (c, _) in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    # 对于一定大小的数字才进行识别
    if w >= 7 and h >= 20:
        # 提取ROI区域
        roi = gray[y:y + h, x:x + w]
        thresh = roi.copy()
        # 智能识别阈值
        T = mahotas.thresholding.otsu(roi)
        thresh[thresh > T] = 255
        # 过滤掉颜色更亮的背景
        thresh = cv2.bitwise_not(thresh)
        # 图片旋转校正，并把数字放到中心
        thresh = deskew(thresh, 20)
        thresh = center_extent(thresh, (20, 20))
        # 测试预处理效果
        cv2.imshow("thresh", thresh)

        # 计算hog算子
        hist = hog.describe(thresh)
        # 根据模型来预测输出
        digit = model.predict([hist])[0]
        print("I think that number is: {}".format(digit))

        # 把识别出的数字用绿色框显示出来
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # 在识别出来的框左上角标注数字
        cv2.putText(image, str(digit), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey(0)
cv2.destroyAllWindows()
