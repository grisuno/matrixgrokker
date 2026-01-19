# MatrixGrokker: Neural Network Matrix Multiplication Learning with Grokking Analysis
**grisun0**
Independent Research
December 2025

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18072859.svg)](https://doi.org/10.5281/zenodo.18072859)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

A PyTorch implementation for studying grokking phenomena in neural networks trained on matrix multiplication tasks, featuring adaptive regularization and zero-shot transfer capabilities.

## Overview

MatrixGrokker is a research framework designed to investigate how neural networks learn matrix multiplication operations, with particular focus on grokking behavior - the phenomenon where networks suddenly transition from memorization to generalization. The implementation includes sophisticated monitoring of learning dynamics through local complexity and superposition metrics, coupled with an adaptive thermal regularization engine.

## Key Features

Grokking Analysis: Comprehensive tracking of the transition from memorization to generalization
Thermal Engine: Adaptive weight decay regulation based on local complexity and superposition metrics
Zero-Shot Transfer: Ability to transfer learned representations to larger matrix sizes without additional training
Comprehensive Metrics: Real-time monitoring of loss, accuracy, local complexity, and superposition measures
Checkpoint System: Robust saving and resuming capabilities for long training runs
Modular Architecture: Extensible design for experimenting with different network configurations

## Installation

```bash
git clone https://github.com/grisuno/matrixgrokker
cd matrixgrokker
pip install torch numpy
```

## Quick Start

Run the complete experiment with default configuration:
```Python

from app import run_full_experiment

grokker, metrics, transfer_results = run_full_experiment()
```

This will:

Train a neural network on 2×2 matrix multiplication
Monitor grokking behavior through local complexity and superposition metrics
Apply adaptive thermal regularization
Test zero-shot transfer to 4×4 and 8×8 matrices
Configuration
The system is configured through the Config class:
```Python
from app import Config

config = Config()
config.MATRIX_SIZE = 2           # Base matrix size for training
config.HIDDEN_DIM = 256          # Hidden layer dimensions
config.NUM_LAYERS = 3            # Network depth
config.TRAIN_EPOCHS = 1000       # Training duration
config.BATCH_SIZE = 128          # Batch size
config.LEARNING_RATE = 0.001     # Learning rate
config.WEIGHT_DECAY = 0.01       # Base weight decay
```

## Architecture
### Core Components
- MLPModel: Multi-layer perceptron with configurable depth and activation functions
1. Supports weight expansion for transfer learning
2. Implements forward hooks for activation analysis
3. Provides weight matrix extraction for superposition analysis
- MatrixMultiplicationDataset: Synthetic data generation
1. Generates random matrix pairs within specified ranges
2. Computes exact multiplication results
3. Flattens matrices for network consumption
- LocalComplexity: Measures neural representation diversity
1. Computes pairwise activation similarities
2. Quantifies the complexity of learned representations
3. Values range from 0 (simple) to 1 (complex)
Superposition: Analyzes weight matrix structure
1. Performs singular value decomposition on weight matrices
2. Measures the degree of weight superposition
3. Values range from 0 (no superposition) to 1 (high superposition)
- ThermalEngine: Adaptive regularization system
1. Adjusts weight decay based on local complexity and superposition
2. Targets optimal learning dynamics
3. Provides thermal progress indicators
4. Training Pipeline
5. The training process implements a sophisticated monitoring system:
6. Forward Pass: Compute predictions and loss
7. Backward Pass: Update network parameters
- Metrics Computation: Calculate local complexity and superposition every N steps
- Thermal Adjustment: Modify weight decay based on current metrics
- Checkpointing: Save model state and metrics periodically
- Validation: Evaluate on held-out validation set
- Transfer Learning
- The zero-shot transfer mechanism allows the network to generalize to larger matrices:
- Weight Expansion: Increase network capacity for larger input/output dimensions
- Zero-Shot Evaluation: Test on larger matrices without additional training
- Performance Analysis: Measure accuracy and learning dynamics on transfer tasks
- Metrics and Monitoring
1. The system tracks comprehensive metrics throughout training:
- Performance Metrics:
1. Training and validation loss (MSE)
2. Training and validation accuracy (within 0.1 threshold)
3. Iterations per second for performance monitoring
- Learning Dynamics:
1. Local Complexity (LC): Measures representation diversity
2. Superposition (SP): Analyzes weight matrix structure
- Thermal Progress: Combined measure of learning optimization
- Regularization:
1. Adaptive weight decay values
2. Learning rate scheduling
3. Thermal engine status

## Appendix C: Reproducibility

Repository: https://github.com/grisuno/strass_strassen

DOI: https://doi.org/10.5281/zenodo.18263654

Reproduction:

```bash
git clone https://github.com/grisuno/strass_strassen
cd strass_strassen
pip install -r requirements.txt
python app.py
```

Related repositories:

- Ancestor: https://github.com/grisuno/SWAN-Phoenix-Rising
- Core Framework: https://github.com/grisuno/agi
- Parity Cassette: https://github.com/grisuno/algebra-de-grok
- Wave Cassette: https://github.com/grisuno/1d_wave_equation_grokker
- Kepler Cassette: https://github.com/grisuno/kepler_orbit_grokker
- Pendulum Cassette: https://github.com/grisuno/chaotic_pendulum_grokked
- Ciclotron Cassette: https://github.com/grisuno/supertopo3
- MatMul 2x2 Cassette: https://github.com/grisuno/matrixgrokker
- HPU Hamiltonian Cassette: https://github.com/grisuno/HPU-Core

[https://zenodo.org/records/18295001](https://zenodo.org/records/18295001)
---

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
