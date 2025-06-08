import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import image_dataset_from_directory


def binarize(img):
    """将图像二值化，增强图像特征，兼容灰度和RGB输入"""
    arr = img_to_array(img, dtype='uint8')
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    binarized = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    inverted_binary_img = ~binarized
    return np.expand_dims(inverted_binary_img, -1)

def build_model(num_classes=14):
    """构建CNN模型
    14类：+ - 0-9 = div times
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
    训练模型函数，适配tf.keras 2.13+，使用image_dataset_from_directory
    """
    # 加载数据集
    train_ds = image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='categorical',
        color_mode='grayscale',
        batch_size=batch_size,
        image_size=(img_height, img_width),
        shuffle=True,
        seed=123
    )
    class_names = train_ds.class_names
    print("训练类别：", class_names)

    # 数据增强和二值化（可选：如需自定义预处理，可用map）
    def preprocess(x, y):
        # x: (batch, h, w, 1)
        x = tf.map_fn(lambda img: tf.numpy_function(binarize, [img], tf.uint8), x, fn_output_signature=tf.TensorSpec(shape=(45, 45, 1), dtype=tf.uint8))
        x = tf.cast(x, tf.float32) / 255.0
        return x, y
    
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # 构建模型
    model = build_model(len(class_names))
    model.summary()

    # 训练模型
    model.fit(
        train_ds,
        epochs=epochs
    )

    # 保存模型
    model.save('calculator/model.keras')
    print("模型已保存到 'calculator/model.keras' 目录")
    return class_names, model

if __name__ == "__main__":
    class_names, model = train_model()
    print(f"模型训练完成，共识别 {len(class_names)} 个类别")
