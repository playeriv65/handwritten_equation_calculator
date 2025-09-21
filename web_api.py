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
from calculator.equation_solver import solve_equation

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

def process_equation(image_path):
    """处理算式图像的通用函数"""
    try:
        # 确保calculator已初始化
        global calculator
        if calculator is None:
            # 使用绝对路径
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calculator', 'model.keras')
            print(f"尝试加载模型: {model_path}")
            calculator = HandwrittenEquationCalculator(model_path)
            print("模型加载成功!")
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            return {
                'success': False,
                'error': f'图像文件不存在: {image_path}'
            }
        
        print(f"处理图像: {image_path}")
            
        # 识别
        equation = calculator.recognize_equation(image_path)
        
        # 求解
        result = solve_equation(equation)
        formatted_result = str(result)
        
        return {
            'success': True,
            'equation': equation,
            'result': formatted_result
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'处理图像时出错: {str(e)}'
        }

@app.route('/api/recognize', methods=['POST'])
def recognize():
    """处理文件上传的识别API接口"""
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
        }), 400    # 生成唯一文件名，避免非ASCII字符问题
    import uuid
    try:
        _, file_extension = os.path.splitext(file.filename)
    except:
        # 如果文件名有问题，使用默认扩展名
        file_extension = '.png'
    safe_filename = f"{uuid.uuid4().hex}{file_extension}"
    
    # 保存上传的文件
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           app.config['UPLOAD_FOLDER'], 
                           safe_filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 先保存文件
    file.save(file_path)
    print(f"原始图像已保存到: {file_path} (原文件名: {file.filename})")
    
    # 处理可能的透明背景问题
    try:
        # 读取已保存的图像，检查是否为透明PNG
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is not None and len(image.shape) > 2 and image.shape[2] == 4:
            print(f"检测到上传的图片含有透明通道，尺寸: {image.shape}")
            # 添加白色背景
            white_background = np.ones_like(image, dtype=np.uint8) * 255
            # 混合前景和白色背景
            alpha_channel = image[:,:,3] / 255.0
            for c in range(3):
                white_background[:,:,c] = (alpha_channel * image[:,:,c] + 
                                        (1-alpha_channel) * white_background[:,:,c])
            # 保存处理后的图像
            cv2.imwrite(file_path, white_background[:,:,:3])
            print(f"已添加白色背景并重新保存")
    except Exception as e:
        print(f"处理上传图像时出错 (但继续处理): {str(e)}")
    
    # 处理图像并返回结果
    print(f"Calculator状态: {calculator}")
    result = process_equation(file_path)
    print(f"处理后Calculator状态: {calculator}")
    return jsonify(result), 500 if not result['success'] else 200

@app.route('/api/recognize-base64', methods=['POST'])
def recognize_base64():
    """处理Base64编码图像的识别API接口"""
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
            
        # 添加调试信息
        print(f"接收到Base64图像数据，长度: {len(image_data)}")
        
        # 解码Base64数据
        try:
            image_bytes = base64.b64decode(image_data)
            print(f"成功解码Base64数据，字节长度: {len(image_bytes)}")
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Base64解码失败: {str(e)}'
            }), 400
            
        # 使用UUID生成唯一的临时文件名
        import uuid
        temp_filename = f"{uuid.uuid4().hex}.png"
        temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                app.config['UPLOAD_FOLDER'], 
                                temp_filename)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)        # 保存为临时文件
        try:
            # 直接使用OpenCV处理图像，避免PIL可能出现的问题
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # 首先尝试读取为带有透明通道的图像
            image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if image is None:
                return jsonify({
                    'success': False,
                    'error': 'Base64图像解码失败，无法转换为图像数据'
                }), 400
                
            # 检查图像通道数，处理透明PNG
            print(f"原始图像尺寸: {image.shape}")
            if len(image.shape) > 2 and image.shape[2] == 4:
                # 有透明通道的图像，创建白色背景
                white_background = np.ones_like(image, dtype=np.uint8) * 255
                # 仅保留RGB通道（去掉Alpha通道）
                alpha_channel = image[:,:,3] / 255.0
                # 混合前景和白色背景
                for c in range(3):
                    white_background[:,:,c] = (alpha_channel * image[:,:,c] + 
                                            (1-alpha_channel) * white_background[:,:,c])
                # 保存RGB图像（无透明通道）
                image = white_background[:,:,:3]
                print("已将透明PNG转换为带白色背景的RGB图像")
            elif len(image.shape) == 2:
                # 灰度图像，转换为RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                print("已将灰度图像转换为RGB")
            
            print(f"处理后图像尺寸: {image.shape}")
            cv2.imwrite(temp_path, image)
            print(f"临时图像已保存到: {temp_path}, 大小: {os.path.getsize(temp_path)}字节")
        except Exception as e:
            print(f"图像处理异常详情: {str(e)}")
            return jsonify({
                'success': False,
                'error': f'图像处理失败: {str(e)}'
            }), 400
        
        # 检查文件是否成功保存
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return jsonify({
                'success': False,
                'error': '临时图像文件保存失败或为空'
            }), 500
        
        # 处理图像并返回结果
        print(f"Calculator状态: {calculator}")
        result = process_equation(temp_path)
        print(f"Base64图像处理结果: {result}")
        return jsonify(result), 500 if not result['success'] else 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'处理图像时出错: {str(e)}'
        }), 500

def create_app(model_path=None):
    """创建Flask应用"""
    global calculator
    
    if model_path is None:
        # 默认模型路径 - 直接指向.keras文件
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calculator', 'model.keras')
    
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
                        help='模型路径 (默认: calculator/model.keras)')
    
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    
    args = parser.parse_args()
    
    app = create_app(args.model)
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
