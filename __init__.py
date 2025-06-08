"""
手写数学算式识别与求解系统

该包含有两个主要模块:
1. calculator: 用于识别和计算手写数学算式
2. train: 用于训练识别模型
"""

from calculator.equation_recognizer import HandwrittenEquationCalculator
from calculator.equation_solver import solve_equation, format_result
