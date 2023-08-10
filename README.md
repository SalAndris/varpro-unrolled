# Unrolled Variable Projection Neural Network

This repository contains a part of the code that I developed for my master's thesis. In my research, I delved into the application of deep unfolding techniques to the Variable Projection method. I explored the essential components required for a successful unfolding process and evaluated the performance of the resulting network on two signal classification tasks. The network achieved **99.8%** accuracy on the synthetic dataset and **94.7%** accuracy on the ECG dataset. For more details you can find the full thesis [here](thesis.pdf).

## Install

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)

```bash
python -m pip install -r requirements.txt
```

## Run

1D function fitting:
```bash
python optimize.py
```

Neural network training:
```bash
python train.py
```

Run type checking:
```bash
mypy . --ignore-missing-imports
```
