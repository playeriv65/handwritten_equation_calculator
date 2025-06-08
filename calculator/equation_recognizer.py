import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# 基本的类别名称，可以根据训练数据进行修改
CLASS_NAMES = ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'div', 'times', 'y']

def binarize(img):
    """将图像二值化，增强图像特征"""
    img = image.img_to_array(img, dtype='uint8')
    binarized = np.expand_dims(cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2), -1)
    inverted_binary_img = ~binarized
    return inverted_binary_img

def getOverlap(a, b):
    """计算两个区间的重叠度"""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def detect_contours(img_path):
    """检测图像中的轮廓"""
    # 读取灰度图像
    input_image = cv2.imread(img_path, 0)
    input_image_cpy = input_image.copy()

    # 将灰度图像二值化，然后反转
    binarized = cv2.adaptiveThreshold(input_image_cpy,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    inverted_binary_img = ~binarized

    # 检测轮廓
    contours_list, hierarchy = cv2.findContours(inverted_binary_img,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    
    # 获取边界框
    l = []
    for c in contours_list:
        x, y, w, h = cv2.boundingRect(c)
        l.append([x, y, w, h])
    
    # 检查是否有重叠的矩形，合并重叠的部分
    lcopy = l.copy()
    keep = []
    while len(lcopy) != 0:
        curr_x, curr_y, curr_w, curr_h = lcopy.pop(0)
        if curr_w * curr_h < 20:  # 移除非常小的框
            continue
        throw = []
        for i, (x, y, w, h) in enumerate(lcopy):
            curr_interval = [curr_x, curr_x+curr_w]
            next_interval = [x, x+w]
            if getOverlap(curr_interval, next_interval) > 1:  # 超过1像素重叠，这是任意的
                # 合并两个区间
                new_interval_x = [min(curr_x, x), max(curr_x+curr_w, x+w)]
                new_interval_y = [min(curr_y, y), max(curr_y+curr_h, y+h)]
                newx, neww = new_interval_x[0], new_interval_x[1] - new_interval_x[0]
                newy, newh = new_interval_y[0], new_interval_y[1] - new_interval_y[0]
                curr_x, curr_y, curr_w, curr_h = newx, newy, neww, newh
                throw.append(i)
        for ind in sorted(throw, reverse=True):
            lcopy.pop(ind)
        keep.append([curr_x, curr_y, curr_w, curr_h])
    
    return keep

def resize_pad(img, size, padColor=255):
    """调整图像大小并添加填充，保持纵横比"""
    h, w = img.shape[:2]
    sh, sw = size

    # 插值方法
    if h > sh or w > sw:  # 缩小图像
        interp = cv2.INTER_AREA
    else:  # 拉伸图像
        interp = cv2.INTER_CUBIC

    # 图像的纵横比
    aspect = w/h

    # 计算缩放和填充大小
    if aspect > 1:  # 水平图像
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # 垂直图像
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # 正方形图像
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # 设置填充颜色
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):  # 彩色图像但只提供了一种颜色
        padColor = [padColor]*3

    # 缩放和填充
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def put_double_asterisk(s):
    """在字母后面的数字前添加双星号，表示指数运算"""
    lst = list(s)
    i = 0
    while i < len(lst)-1:
        if lst[i].isalpha():
            if lst[i+1].isdigit():
                lst.insert(i+1, '**')
                i += 1
        i += 1
    s_new = ''.join(lst)
    return s_new

def put_single_asterisk(s):
    """在数字后面的字母前添加星号，表示乘法运算"""
    lst = list(s)
    i = 0
    while i < len(lst)-1:
        if lst[i].isdigit() and lst[i+1].isalpha():
            lst.insert(i+1, '*')
        i += 1
    s_new = ''.join(lst)
    return s_new

class HandwrittenEquationCalculator:
    def __init__(self, model_path='model'):
        """初始化手写算式识别计算器"""
        self.model_path = model_path
        self.model = None
        self.class_names = CLASS_NAMES
        self.load_model()

    def load_model(self):
        """加载模型"""
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print(f"模型已从 {self.model_path} 成功加载")
        except Exception as e:
            print(f"加载模型时出错: {e}")
            raise

    def recognize_equation(self, img_path):
        """识别图像中的算式"""
        print(f"正在识别图像: {os.path.basename(img_path)}")
        
        # 检测图像中的轮廓
        contours = detect_contours(img_path)
        
        # 从图像中读取数据
        input_image = cv2.imread(img_path, 0)
        inverted_binary_img = binarize(input_image)
        
        # 预测每个轮廓中的符号
        eqn_list = []
        for (x, y, w, h) in sorted(contours, key=lambda x: x[0]):
            # 提取ROI并调整大小
            img = resize_pad(inverted_binary_img[y:y+h, x:x+w], (45, 45), 0)
            
            # 准备进行预测
            first = tf.expand_dims(img, 0)
            second = tf.expand_dims(first, -1)
            predicted = self.model.predict(second, verbose=0)
            max_arg = np.argmax(predicted)
            
            # 获取预测的类别
            pred_class = self.class_names[max_arg]
            
            # 符号转换
            if pred_class == "times":
                pred_class = "*"
            if pred_class == "div":
                pred_class = "/"
                
            eqn_list.append(pred_class)
        
        # 构建算式
        eqn = "".join(eqn_list)
        print(f"识别的算式: {eqn}")
        
        # 格式化算式以便计算
        equation = put_double_asterisk(eqn)
        equation = put_single_asterisk(equation)
        
        return equation

def test_calculator(image_path):
    """测试手写算式识别功能"""
    calculator = HandwrittenEquationCalculator()
    equation = calculator.recognize_equation(image_path)
    print(f"格式化后的算式: {equation}")
    return equation

if __name__ == "__main__":
    # 示例用法
    image_path = "../equation_images/test7.png"  # 替换为你的算式图像路径
    result = test_calculator(image_path)
    print(f"最终算式: {result}")
