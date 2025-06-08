import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

def binarize(img):
    """将图像二值化，增强图像特征"""
    img = image.img_to_array(img, dtype='uint8')
    binarized = np.expand_dims(cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2), -1)
    inverted_binary_img = ~binarized
    return inverted_binary_img

def build_model(num_classes=16):
    """构建CNN模型"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input((45, 45, 1)),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
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
    """训练模型函数"""
    # 数据生成器，对图像进行预处理
    train_datagen = ImageDataGenerator(preprocessing_function=binarize)
    
    # 从目录中加载数据
    train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode="grayscale",
            class_mode="categorical",
            seed=123)
    
    # 获取类名
    class_names = [k for k,v in train_generator.class_indices.items()]
    print("训练类别：", class_names)
    
    # 构建模型
    model = build_model(len(class_names))
    model.summary()
    
    # 训练模型
    model.fit(
        train_generator,
        epochs=epochs
    )
    
    # 保存模型
    model.save('../calculator/model')
    print("模型已保存到 '../calculator/model' 目录")
    
    return class_names, model

if __name__ == "__main__":
    class_names, model = train_model()
    print(f"模型训练完成，共识别 {len(class_names)} 个类别")
