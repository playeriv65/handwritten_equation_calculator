import os
import sys
import argparse
from calculator.equation_recognizer import HandwrittenEquationCalculator
from calculator.equation_solver import solve_equation, format_result

class HandwrittenMathSolver:
    """手写算式识别与求解系统"""
    def __init__(self, model_path=None):
        # 如果未提供模型路径，使用默认路径
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'calculator', 'model')
        
        # 初始化识别器
        self.recognizer = HandwrittenEquationCalculator(model_path)

    def process_image(self, image_path):
        """处理图像并求解算式"""
        # 检查文件是否存在
        if not os.path.exists(image_path):
            return f"错误: 文件不存在 - {image_path}"
        
        try:
            # 识别算式
            equation = self.recognizer.recognize_equation(image_path)
            
            # 求解算式
            result = solve_equation(equation)
            
            # 格式化结果
            formatted_result = format_result(result)
            
            return {
                "equation": equation,
                "result": formatted_result
            }
        except Exception as e:
            return f"处理图像时出错: {str(e)}"

def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='手写数学算式识别与求解')
    parser.add_argument('image_path', help='手写算式图像的路径')
    parser.add_argument('--model', help='模型路径 (可选)', default=None)
    
    args = parser.parse_args()
    
    solver = HandwrittenMathSolver(args.model)
    result = solver.process_image(args.image_path)
    
    if isinstance(result, dict):
        print(f"识别的算式: {result['equation']}")
        print(f"计算结果: {result['result']}")
    else:
        print(result)

if __name__ == "__main__":
    main()
