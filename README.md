# Closed-Form Diffusion Models

PyTorch implementation of ["Closed-Form Diffusion Models"](https://arxiv.org/abs/2310.12395) by Christopher Scarvelis, Haitz Sáez de Ocáriz Borde, and Justin Solomon.

![image](https://github.com/user-attachments/assets/8449b428-17d8-403e-8323-c196bb347bde)


## Overview

This repository implements a novel approach to score-based generative models (SGMs) that generates diverse samples without requiring neural network training. Unlike traditional SGMs that approximate score functions using neural networks, this method uses a closed-form score function with explicit smoothing, enabling efficient sampling on consumer-grade CPUs.

```bash
# Clone the repository
git clone https://github.com/DanJbk/closedformeddiff.git
cd closedformeddiff

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
follow the examples at the examples folder.

### parameters:
* z_0: Inital distribution.
* steps: Number of steps to train on
* x: Dataset in the shape of [Number of samples, dimensions].
* sigma: Smoothing parameter.
* M: Number of perturbations applied each steps.
