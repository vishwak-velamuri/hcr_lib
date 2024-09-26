# HCR-KAN Library

The **HCR-KAN Library** is a Python package designed for hierarchical clustering and knowledge-aware networks. This library provides tools for data normalization, estimation, density analysis, and visualization, as well as utilities for neural network layers and models.

## Directory Structure

```
hcr_kan_lib/
│
├── hcr/
│   ├── __init__.py
│   ├── normalization.py
│   ├── basis.py
│   ├── estimation.py
│   ├── density.py
│   └── visualization.py
│
├── kan/
│   ├── __init__.py
│   ├── model.py
│   └── graph_utils.py
│
├── nn/
│   ├── __init__.py
│   ├── hcr_layer.py
│   └── hcr_network.py
│
├── utils/
│   ├── __init__.py
│   └── data_processing.py
│
├── examples/
│   ├── hcr_mnist.py
│   └── kan_mnist.py
│
├── tests/
│   ├── test_hcr.py
│   ├── test_kan.py
│   └── test_nn.py
│
├── setup.py
├── README.md
└── requirements.txt

```