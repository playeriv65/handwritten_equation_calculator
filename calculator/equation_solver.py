import sympy as sym
from sympy import symbols, sympify, solve, solveset

def solve_equation(equation):
    """
    求解数学方程
    :param equation: 字符串表示的数学方程
    :return: 方程的解
    """
    try:
        # 处理包含x变量的方程
        if 'x' in equation and '=' in equation:
            x = sym.symbols('x')
            left, right = equation.split("=")
            eq = left + '-' + right
            result = sym.solve(eq, (x))
            return result
            
        elif 'x' in equation and '=' not in equation:
            x = sym.symbols('x')
            result = sym.solveset(equation, x)
            return result
        
        # 处理包含y变量的方程
        elif 'y' in equation and '=' in equation:
            y = sym.symbols('y')
            left, right = equation.split("=")
            eq = left + '-' + right
            result = sym.solve(eq, (y))
            return result
            
        elif 'y' in equation and '=' not in equation:
            y = sym.symbols('y')
            result = sym.solveset(equation, y)
            return result
        
        # 处理包含x和y变量的方程
        elif 'x' in equation and 'y' in equation and '=' in equation:
            x, y = sym.symbols('x,y')
            left, right = equation.split("=")
            eq = left + '-' + right
            result = sym.solve(eq, (x, y))
            return result
        
        # 处理三角函数方程
        elif '=' in equation and ('sin' in equation or 'tan' in equation or 'cos' in equation):
            if 'x' in equation and 'y' in equation:
                x, y = symbols('x,y')
                eq_sympy = sympify(equation)
                result = solve((eq_sympy), (x, y))
                return result
            elif 'x' in equation:
                x = symbols('x')
                eq_sympy = sympify(equation)
                result = solve((eq_sympy), x)
                return result
            elif 'y' in equation:
                y = symbols('y')
                eq_sympy = sympify(equation)
                result = solve((eq_sympy), y)
                return result

        # 处理表达式求值
        if "=" not in equation:
            if 'sin' in equation or 'tan' in equation or 'cos' in equation:
                eq = sympify(equation)
                result = eq.evalf()
                return result
            
            # 简化表达式
            result = sym.simplify(equation)
            return result
        else:
            # 处理方程求解
            left, right = equation.split("=")
            result = sym.solve(left, right)

            if len(result) == 0:
                return "无解"
            else:
                return result

    except Exception as e:
        return f"错误: 方程预测或求解错误 ({str(e)})"

def format_result(result):
    """
    格式化计算结果
    :param result: 计算结果
    :return: 格式化后的字符串
    """
    if isinstance(result, list):
        if len(result) == 0:
            return "无解"
        formatted = []
        for sol in result:
            if isinstance(sol, tuple):
                formatted.append(f"({', '.join(str(val) for val in sol)})")
            else:
                formatted.append(str(sol))
        return ", ".join(formatted)
    else:
        return str(result)

def test_equation_solver():
    """测试方程求解器功能"""
    test_cases = [
        "2+3",
        "x+5=10",
        "x**2+2*x+1=0",
        "y=2*x+3",
        "sin(x)=0.5"
    ]
    
    for eq in test_cases:
        print(f"方程: {eq}")
        result = solve_equation(eq)
        formatted = format_result(result)
        print(f"解: {formatted}\n")

if __name__ == "__main__":
    test_equation_solver()
