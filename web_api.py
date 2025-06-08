#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Web服务器API，用于提供手写算式识别与求解服务
"""

from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import sys
import argparse

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calculator.equation_recognizer import HandwrittenEquationCalculator
from calculator.equation_solver import solve_equation, format_result

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 初始化识别器
calculator = None

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/recognize', methods=['POST'])
def recognize():
    """识别API接口"""
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': '没有提供图像文件'
        }), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': '没有选择文件'
        }), 400
    
    try:
        # 保存上传的文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # 识别方程
        equation = calculator.recognize_equation(file_path)
        
        # 求解方程
        result = solve_equation(equation)
        formatted_result = format_result(result)
        
        return jsonify({
            'success': True,
            'equation': equation,
            'result': formatted_result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'处理图像时出错: {str(e)}'
        }), 500

@app.route('/api/recognize-base64', methods=['POST'])
def recognize_base64():
    """接收Base64编码的图像并识别"""
    data = request.json
    if not data or 'image' not in data:
        return jsonify({
            'success': False,
            'error': '没有提供Base64图像数据'
        }), 400
    
    try:
        # 解码Base64图像
        image_data = data['image']
        if 'data:image/' in image_data:  # 处理可能的数据URL
            image_data = image_data.split(',')[1]
            
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        
        # 保存临时文件
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp.png')
        image.save(temp_path)
        
        # 识别方程
        equation = calculator.recognize_equation(temp_path)
        
        # 求解方程
        result = solve_equation(equation)
        formatted_result = format_result(result)
        
        return jsonify({
            'success': True,
            'equation': equation,
            'result': formatted_result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'处理图像时出错: {str(e)}'
        }), 500

def create_app(model_path=None):
    """创建Flask应用"""
    global calculator
    
    if model_path is None:
        # 默认模型路径
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'calculator', 'model')
    
    try:
        calculator = HandwrittenEquationCalculator(model_path)
        print(f"模型已从 {model_path} 成功加载")
    except Exception as e:
        print(f"警告: 加载模型失败 - {e}")
        print("API将在有请求时尝试加载模型")
    
    return app

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='启动手写算式识别与求解Web服务')
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='服务器主机地址 (默认: 0.0.0.0)')
    
    parser.add_argument('--port', type=int, default=5000,
                        help='服务器端口 (默认: 5000)')
    
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径 (默认: ../calculator/model)')
    
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    
    args = parser.parse_args()
    
    app = create_app(args.model)
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
