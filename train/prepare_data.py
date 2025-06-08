#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据准备工具，用于准备训练数据集
"""

import os
import sys
import shutil
import argparse
import cv2
import numpy as np
from pathlib import Path

def binarize_image(img):
    """将图像二值化"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return ~binarized  # 反转图像，使字符为白色，背景为黑色

def resize_pad_image(img, target_size=(45, 45), pad_color=0):
    """调整图像大小并添加填充，保持纵横比"""
    h, w = img.shape[:2]
    sh, sw = target_size
    
    # 图像的纵横比
    aspect = w/h
    
    # 计算缩放和填充大小
    if aspect > 1:  # 水平图像
        new_w = sw
        new_h = int(new_w/aspect)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = int(np.floor(pad_vert)), int(np.ceil(pad_vert))
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # 垂直图像
        new_h = sh
        new_w = int(new_h*aspect)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = int(np.floor(pad_horz)), int(np.ceil(pad_horz))
        pad_top, pad_bot = 0, 0
    else:  # 正方形图像
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    
    # 缩放和填充
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if h > sh else cv2.INTER_CUBIC)
    padded_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, 
                                    borderType=cv2.BORDER_CONSTANT, value=pad_color)
    
    return padded_img

def prepare_dataset(input_dir, output_dir, target_size=(45, 45), binarize=True):
    """准备训练数据集"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 记录处理了多少个文件
    processed_count = 0
    
    # 遍历输入目录中的所有类别文件夹
    for class_dir in input_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        print(f"处理类别: {class_name}")
        
        # 创建对应的输出类别文件夹
        output_class_path = output_path / class_name
        output_class_path.mkdir(exist_ok=True)
        
        # 处理该类别下的所有图片
        for img_path in class_dir.glob("*.*"):
            if not img_path.is_file():
                continue
            
            # 检查是否是图像文件
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                continue
            
            try:
                # 读取图像
                img = cv2.imread(str(img_path))
                
                # 确保图像加载成功
                if img is None:
                    print(f"警告: 无法读取图像 {img_path}")
                    continue
                
                # 二值化处理
                if binarize:
                    img = binarize_image(img)
                
                # 调整大小和添加填充
                img = resize_pad_image(img, target_size)
                
                # 保存处理后的图像
                output_img_path = output_class_path / img_path.name
                cv2.imwrite(str(output_img_path), img)
                
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"已处理 {processed_count} 个图像...")
                
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
    
    print(f"数据集准备完成，共处理 {processed_count} 个图像")

def main():
    """命令行工具入口函数"""
    parser = argparse.ArgumentParser(description='准备手写算式识别的训练数据集')
    
    parser.add_argument('--input-dir', type=str, required=True,
                        help='输入数据集目录路径，包含分类的子文件夹')
    
    parser.add_argument('--output-dir', type=str, default='dataset',
                        help='处理后的数据集输出目录 (默认: dataset)')
    
    parser.add_argument('--img-size', type=int, default=45,
                        help='处理后图像的尺寸 (默认: 45)')
    
    parser.add_argument('--no-binarize', action='store_true',
                        help='不对图像进行二值化处理')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入目录 '{args.input_dir}' 不存在。")
        return 1
    
    try:
        prepare_dataset(
            args.input_dir, 
            args.output_dir, 
            target_size=(args.img_size, args.img_size),
            binarize=not args.no_binarize
        )
        return 0
    
    except Exception as e:
        print(f"处理数据集时出错: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
