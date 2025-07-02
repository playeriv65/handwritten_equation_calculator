# 手写数学算式识别与求解系统

这个项目实现了基于CNN的手写数学算式识别与求解系统，可以识别图像中的手写算式并进行计算求解。系统支持加减乘除四则运算，并可以求解简单的方程。

## 主要功能

- 手写数学算式识别
- 数学表达式计算
- 方程求解
- 支持Web界面和API调用
- 易于扩展的模型训练框架

## 项目结构

```
handwritten_equation_calculator/
├── calculator/                  # 算式识别与求解模块
│   ├── __init__.py              # 包初始化文件
│   ├── equation_recognizer.py   # 算式识别器
│   ├── equation_solver.py       # 算式求解器
│   └── model/                   # 训练好的模型（需要自行训练）
├── train/                       # 模型训练模块
│   ├── __init__.py              # 包初始化文件
│   ├── model_trainer.py         # 模型训练器
│   ├── train_cmd.py             # 命令行训练工具
│   ├── prepare_data.py          # 数据准备工具
│   └── dataset/                 # 训练数据集（需要自行准备）
│       ├── +/                   # 加号样本
│       ├── -/                   # 减号样本
│       ├── */                   # 乘号样本
│       └── .../                 # 其他符号样本
├── templates/                   # Web界面模板
│   └── index.html               # 主页模板
├── uploads/                     # 上传文件临时存储
├── examples/                    # 示例图像
├── __init__.py                  # 包初始化文件
├── main.py                      # 主程序入口
├── web_api.py                   # Web服务API
├── requirements.txt             # 项目依赖
├── USAGE.md                     # 详细使用说明
└── README.md                    # 项目说明
```

## 支持的符号

本系统可以识别以下符号：
- 数字：0-9
- 运算符：+（加）, -（减）, *（乘）, /（除法）
- 等号：=

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备并处理训练数据

```bash
python train/prepare_data.py --input-dir raw_data --output-dir train/dataset
```

### 3. 训练模型

```bash
python train/train_cmd.py --data-dir train/dataset --epochs 20 --batch-size 32
```

训练完成后，命令行将输出验证集上的分类报告（per-class 精度/召回/F1）以及 Micro/Macro 平均指标。

### 4. 识别并求解手写算式

```bash
python main.py path/to/your/equation/image.png
```

### 5. 启动Web服务

```bash
python web_api.py
```

访问 http://localhost:5000 使用Web界面。

## 示例

命令行使用：
```
> python main.py examples/equation.png
识别的算式: 3+4=7
计算结果: True
```

API调用：
```python
import requests

with open('equation.png', 'rb') as f:
    response = requests.post('http://localhost:5000/api/recognize', files={'image': f})
    result = response.json()
    print(f"算式: {result['equation']}, 结果: {result['result']}")
```

## 详细文档

更多详细的安装和使用说明，请参阅 [USAGE.md](USAGE.md) 文件。

## 许可证

MIT
