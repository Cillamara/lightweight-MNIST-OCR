# lightweight-MNIST-OCR
Logistic regression based OCR for MNIST dataset

## Setup & Build
### 1. Dependencies
- Python 3
- CMake
- pybind11

```bash
pip install pybind11
pip install torch torchvision
```
### 2. Build Project
```
mkdir build
cd build
cmake ..
make
```

### 3. Usage
```
import sys
sys.path.append('../bin')

import mnistocr
```

  
