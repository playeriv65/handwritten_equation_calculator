# Handwritten Equation Recognition and Solver System

This project implements a CNN-based system for recognizing and solving handwritten mathematical equations from images. It supports basic arithmetic operations and simple equation solving, and provides both a web interface and API for easy integration.

## Features

- Handwritten mathematical equation recognition
- Mathematical expression calculation
- Simple equation solving
- Web interface and API support
- Extensible model training framework

## Project Structure

```
handwritten_equation_calculator/
├── calculator/                  # Recognition and solving modules
│   ├── __init__.py
│   ├── equation_recognizer.py   # Equation recognizer
│   ├── equation_solver.py       # Equation solver
│   ├── model.keras              # Trained model (to be trained by user)
│   └── ...
├── train/                       # Model training modules
│   ├── __init__.py
│   ├── model_trainer.py         # Model trainer
│   ├── train_cmd.py             # CLI training tool
│   ├── prepare_data.py          # Data preparation tool
│   └── dataset/                 # Training dataset (to be prepared by user)
├── templates/                   # Web interface templates
│   └── index.html
├── uploads/                     # Temporary storage for uploads
├── examples/                    # Example images
├── __init__.py
├── main.py                      # Main entry point
├── web_api.py                   # Web service API
├── requirements.txt             # Project dependencies
├── USAGE.md                     # Detailed usage instructions
└── README.md                    # Project documentation
```

## Supported Symbols

The system can recognize the following symbols:
- Digits: 0-9
- Operators: + (add), - (subtract), * (multiply), / (divide)
- Equal sign: =

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare and Process Training Data

```bash
python train/prepare_data.py --input-dir raw_data --output-dir train/dataset
```

### 3. Train the Model

```bash
python train/train_cmd.py --data-dir train/dataset --epochs 20 --batch-size 32
```

After training, the command line will output a classification report (per-class precision/recall/F1) and micro/macro averages for the validation set.

### 4. Recognize and Solve Handwritten Equations

```bash
python main.py path/to/your/equation/image.png
```

### 5. Start the Web Service

```bash
python web_api.py
```

Visit http://localhost:5000 to use the web interface.

## Examples

Command line usage:
```
> python main.py examples/equation.png
Recognized equation: 3+4=7
Result: True
```

API usage:
```python
import requests

with open('equation.png', 'rb') as f:
    response = requests.post('http://localhost:5000/api/recognize', files={'image': f})
    result = response.json()
    print(f"Equation: {result['equation']}, Result: {result['result']}")
```

## Documentation

For more detailed installation and usage instructions, please refer to the [USAGE.md](USAGE.md) file.

## License

MIT
