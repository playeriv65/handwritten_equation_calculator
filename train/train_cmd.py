#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
命令行训练工具，用于训练手写算式识别模型
"""

import argparse
import os
import sys
from model_trainer import train_model

def main():
    """命令行工具入口函数"""
    parser = argparse.ArgumentParser(description='训练手写算式识别模型')
    
    parser.add_argument('--data-dir', type=str, default='dataset',
                        help='训练数据集目录路径 (默认: dataset)')
    
    parser.add_argument('--batch-size', type=int, default=32,
                        help='训练批次大小 (默认: 32)')
    
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数 (默认: 10)')
    
    parser.add_argument('--img-height', type=int, default=45,
                        help='图像高度 (默认: 45)')
    
    parser.add_argument('--img-width', type=int, default=45,
                        help='图像宽度 (默认: 45)')
    
    parser.add_argument('--output-dir', type=str, default='../calculator/model',
                        help='模型保存目录 (默认: ../calculator/model)')
    
    args = parser.parse_args()
    
    # 检查数据集目录是否存在
    if not os.path.isdir(args.data_dir):
        print(f"错误: 数据集目录 '{args.data_dir}' 不存在。")
        return 1
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    
    print(f"开始训练模型，数据集目录: {args.data_dir}")
    print(f"批次大小: {args.batch_size}, 训练轮数: {args.epochs}")
    print(f"图像尺寸: {args.img_height}x{args.img_width}")
    
    try:
        # 训练模型
        class_names, model = train_model(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            img_height=args.img_height,
            img_width=args.img_width,
            epochs=args.epochs
        )
        
        # 保存模型到指定目录
        os.makedirs(args.output_dir, exist_ok=True)
        model.save(args.output_dir)
        
        print(f"模型训练完成，已保存到: {args.output_dir}")
        print(f"模型可识别 {len(class_names)} 个类别: {', '.join(class_names)}")
        return 0
    
    except Exception as e:
        print(f"训练过程中出错: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
