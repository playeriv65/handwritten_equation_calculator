# 安装与使用说明

## 1. 环境要求
- Python 3.6 或以上
- TensorFlow 2.0 以上
- OpenCV, NumPy, Matplotlib, Flask 等依赖库

## 2. 安装依赖

```bash
pip install tensorflow opencv-python numpy matplotlib sympy pillow flask
```

或者使用提供的 requirements.txt 文件：

```bash
pip install -r requirements.txt
```

## 3. 数据集准备

### 3.1 准备原始数据
为每个符号（加号、减号等）创建单独的文件夹，并收集相应的样本图像。例如：

```
raw_data/
  +/
    img1.png
    img2.png
    ...
  -/
    img1.png
    img2.png
    ...
  ...
```

### 3.2 处理数据集
使用数据处理工具预处理数据：

```bash
python train/prepare_data.py --input-dir raw_data --output-dir train/dataset
```

## 4. 训练模型

### 4.1 使用命令行工具训练
```bash
python train/train_cmd.py --data-dir train/dataset --epochs 20 --batch-size 32
```

训练完成后，命令行将输出验证集上的分类报告（per-class 精度/召回/F1）以及 Micro/Macro 平均指标。

### 4.2 使用 Jupyter Notebook 训练
也可以使用提供的 demo.ipynb 文件进行训练。

## 5. 使用模型

### 5.1 命令行使用
```bash
python main.py path/to/your/equation/image.png
```

### 5.2 启动Web服务
```bash
python web_api.py
```
然后访问 http://localhost:5000 使用Web界面。

### 5.3 API调用
启动Web服务后，可以通过以下API调用：

- POST /api/recognize - 上传图像文件
- POST /api/recognize-base64 - 提交Base64编码的图像

## 6. 示例用法

### 命令行示例
```bash
python main.py examples/equation.png
```

### API调用示例
```python
import requests

# 上传图像
with open('equation.png', 'rb') as f:
    response = requests.post('http://localhost:5000/api/recognize', files={'image': f})
    result = response.json()
    print(result)

# 使用Base64
import base64
with open('equation.png', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
    response = requests.post('http://localhost:5000/api/recognize-base64', json={'image': image_data})
    result = response.json()
    print(result)
```

## 7. 支持的符号
- 数字：0-9
- 运算符：+, -, *, /（除法）
- 等号：=
- 变量：y
