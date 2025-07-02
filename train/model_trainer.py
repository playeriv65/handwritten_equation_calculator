import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 减少日志
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
import sys

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def binarize(img):
    """将图像二值化，增强图像特征，兼容灰度和RGB输入"""
    arr = img_to_array(img, dtype='uint8')
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    _, binarized = cv2.threshold(arr, 128, 255, cv2.THRESH_BINARY)
    inverted_binary_img = binarized
    return np.expand_dims(inverted_binary_img, -1)

def build_model(num_classes=14):
    """构建CNN模型
    类别：+ - 0-9 = div times
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input((45, 45, 1)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    
    model.compile(
        optimizer='adam',
        loss=tf.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    return model

def train_model(data_dir='dataset', batch_size=32, img_height=45, img_width=45, epochs=3):
    """
    训练模型函数
    """
    CLASS_NAMES = ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'div', 'times']
    
    # 验证数据集中的类别是否都存在
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            raise ValueError(f"找不到类别 {class_name} 的训练数据目录: {class_dir}")
    
    # 手动构建文件列表和标签，实现分层抽样
    file_paths, labels = [], []
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            path = os.path.join(class_dir, fname)
            if os.path.isfile(path):
                file_paths.append(path)
                labels.append(idx)
    # stratified split，确保每个类别都有验证样本
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=123
    )
    # 构建 tf.data 数据集
    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=1)
        img = tf.numpy_function(binarize, [img], tf.uint8)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize_with_pad(img, img_height, img_width)
        return img, tf.one_hot(label, len(CLASS_NAMES))
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(len(train_paths), seed=123).map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 构建模型
    model = build_model(len(CLASS_NAMES))
    model.summary()

    # 训练模型
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # 保存模型
    model.save('calculator/model.keras')
    print("模型已保存到 'calculator/model.keras' 目录")
    # 在验证集上进行评估
    val_labels = np.vstack([y.numpy() for _, y in val_ds])
    preds = model.predict(val_ds, verbose=0)
    y_true = np.argmax(val_labels, axis=1)
    y_pred = np.argmax(preds, axis=1)
    # 打印每个类别的指标报告
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\nValidation Classification Report:\n", report)
    # 计算并打印微平均和宏平均指标
    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print(f"Validation Micro -> Precision: {micro_p:.4f}, Recall: {micro_r:.4f}, F1: {micro_f:.4f}")
    print(f"Validation Macro -> Precision: {macro_p:.4f}, Recall: {macro_r:.4f}, F1: {macro_f:.4f}")
    return CLASS_NAMES, model

if __name__ == "__main__":
    class_names, model = train_model()
    print(f"模型训练完成，共识别 {len(class_names)} 个类别")
