import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import img_to_array

# 数字和四则运算符类别
CLASS_NAMES = ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'div', 'times']

def binarize(img):
    """将图像二值化，增强图像特征，兼容灰度和RGB输入"""
    arr = img_to_array(img, dtype='uint8')
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    # 使用自适应阈值进行二值化，与训练时保持一致
    _, binarized = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY)
    return np.expand_dims(binarized, -1)

def getOverlap(a, b):
    """计算两个区间的重叠度"""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def detect_contours(img_path, debug_dir, base_file_name):
    """检测图像中的轮廓并保存中间步骤"""
    # 读取灰度图像
    input_image = cv2.imread(img_path, 0)
    input_image_cpy = input_image.copy()
    cv2.imwrite(os.path.join(debug_dir, f"{base_file_name}_01_grayscale.png"), input_image_cpy)

    # 将灰度图像二值化，然后反转
    binarized = cv2.adaptiveThreshold(input_image_cpy,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    cv2.imwrite(os.path.join(debug_dir, f"{base_file_name}_02_binarized.png"), binarized)
    inverted_binary_img = ~binarized
    cv2.imwrite(os.path.join(debug_dir, f"{base_file_name}_03_inverted_binary_for_contours.png"), inverted_binary_img)

    # 检测轮廓
    contours_list, hierarchy = cv2.findContours(
        inverted_binary_img,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 在彩色副本上绘制所有轮廓
    img_with_all_contours = cv2.cvtColor(input_image_cpy, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_with_all_contours, contours_list, -1, (0, 255, 0), 1) # 绿色绘制轮廓
    cv2.imwrite(os.path.join(debug_dir, f"{base_file_name}_04_all_contours_drawn.png"), img_with_all_contours)
    
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
    
    # 在彩色副本上绘制合并后的轮廓
    img_with_merged_contours = cv2.cvtColor(input_image_cpy, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in keep:
        cv2.rectangle(img_with_merged_contours, (x, y), (x+w, y+h), (0, 0, 255), 1) # 红色绘制矩形
    cv2.imwrite(os.path.join(debug_dir, f"{base_file_name}_05_merged_contours_drawn.png"), img_with_merged_contours)
    
    return keep

def resize_pad(img, size, padColor=255):
    """调整图像大小并添加填充，保持纵横比"""
    h, w = img.shape[:2]
    sh, sw = size

    interp = cv2.INTER_AREA

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
        debug_dir = 'debug_split'
        os.makedirs(debug_dir, exist_ok=True)
        
        base_img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # 首先检查图像是否存在
        if not os.path.exists(img_path):
            print(f"错误: 图像路径不存在: {img_path}")
            return ""
            
        # 尝试读取图像，包括可能的透明通道
        try:
            # 先尝试读取带透明通道的版本
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"警告: 无法读取图像: {img_path}")
                return ""
                
            # 检查并处理透明通道
            if len(image.shape) > 2 and image.shape[2] == 4:
                print(f"检测到透明通道，进行处理...")
                # 有透明通道，添加白色背景
                white_background = np.ones_like(image, dtype=np.uint8) * 255
                alpha_channel = image[:,:,3] / 255.0
                for c in range(3):
                    white_background[:,:,c] = (alpha_channel * image[:,:,c] + 
                                              (1-alpha_channel) * white_background[:,:,c])
                # 转为灰度
                input_image_gray = cv2.cvtColor(white_background[:,:,:3], cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # 标准RGB图像，转为灰度
                input_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                # 已经是灰度图像
                input_image_gray = image
                
            # 保存处理后的灰度图像用于调试
            cv2.imwrite(os.path.join(debug_dir, f"{base_img_name}_01_grayscale.png"), input_image_gray)
        except Exception as e:
            print(f"读取图像时出错: {str(e)}")
            # 回退到标准灰度读取方式
            input_image_gray = cv2.imread(img_path, 0)

        # 检测图像中的轮廓
        contours = detect_contours(img_path, debug_dir, base_img_name)
        
        # 增加反色处理，变成白底黑字 (用于ROI提取)
        input_image_inverted_for_roi = 255 - input_image_gray
        cv2.imwrite(os.path.join(debug_dir, f"{base_img_name}_06_inverted_input_for_roi.png"), input_image_inverted_for_roi)
        
        eqn_list = []
        for idx, (x, y, w, h) in enumerate(sorted(contours, key=lambda x: x[0])):
            # 从反色后的图像中提取ROI
            char_roi_original = input_image_inverted_for_roi[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(debug_dir, f"{base_img_name}_roi_{idx:02d}_A_original_patch_x{x}y{y}.png"), char_roi_original)

            # 先分割和resize
            roi_resized_padded = resize_pad(char_roi_original, (45, 45), 0) # 使用黑色填充
            cv2.imwrite(os.path.join(debug_dir, f"{base_img_name}_roi_{idx:02d}_B_resized_padded.png"), roi_resized_padded)
            
            # 二值化
            roi_binarized = binarize(roi_resized_padded)
            cv2.imwrite(os.path.join(debug_dir, f"{base_img_name}_roi_{idx:02d}_C_binarized.png"), roi_binarized)
            
            # 预测
            first = tf.expand_dims(roi_binarized, 0)
            predicted = self.model.predict(first, verbose=0)
            
            # 打印概率分布，帮助调试
            probabilities = tf.nn.softmax(predicted[0]).numpy()
            print(f"\n字符 {idx} 的预测概率分布:")
            for cls_name, prob in zip(self.class_names, probabilities):
                print(f"{cls_name}: {prob:.3f}", end=" ")
            print()  # 换行
            
            max_arg = np.argmax(predicted)
            pred_class = self.class_names[max_arg]
            if pred_class == "times":
                pred_class = "*"
            if pred_class == "div":
                pred_class = "/"
            eqn_list.append(pred_class)
        eqn = "".join(eqn_list)
        print(f"识别的算式: {eqn}")
        return eqn

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
