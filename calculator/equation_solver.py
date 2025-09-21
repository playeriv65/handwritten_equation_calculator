import sympy as sym
from sympy import sympify

def solve_equation(equation):
    """
    只支持数字四则运算和等式求值
    :param equation: 字符串表示的数学算式
    :return: 计算结果
    """
    try:
        allowed_chars = set('0123456789+-*/.= ')
        if not set(equation).issubset(allowed_chars):
            return "仅支持数字和四则运算"
        
        if '=' in equation:
            left, right = equation.split('=')
            try:
                left_val = sympify(left)
                right_val = sympify(right)
                return bool(left_val == right_val)
            except Exception as e:
                return f"错误: 等式求值失败 ({str(e)})"
        else:
            try:
                result = sympify(equation).evalf()
                return result
            except Exception as e:
                return f"错误: 表达式求值失败 ({str(e)})"
    except Exception as e:
        return f"错误: 求解失败 ({str(e)})"

def test_equation_solver():
    """测试方程求解器功能"""
    test_cases = [
        "2+3",
        "8-5*2",
        "6/2+1",
        "3+4=7",
        "10-2=8",
        "2*3=7"
    ]
    
    for eq in test_cases:
        print(f"算式: {eq}")
        result = solve_equation(eq)
        formatted = str(result)
        print(f"结果: {formatted}\n")

if __name__ == "__main__":
    test_equation_solver()
