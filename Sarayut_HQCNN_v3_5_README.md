# Sarayut-HQCNN v3.5

Hybrid Quantum--Classical Stability Framework

------------------------------------------------------------------------

## Overview

Sarayut-HQCNN v3.5 is a stability-aware hybrid optimization framework
designed for quantum-classical machine learning systems.

The framework integrates a Stability Operator (Σ) regularization
mechanism to reduce sensitivity to parameter perturbations and improve
robustness under noise conditions.

------------------------------------------------------------------------

## Core Concept

Stability Operator:

Σ(θ) = E\[(f(θ + ε) − f(θ))²\]

Where: - θ = trainable parameters - ε = small perturbation noise - f(·)
= model output function

The total loss is defined as:

L_total = L_task + λ Σ(θ)

This encourages flat minima and improved robustness.

------------------------------------------------------------------------

## Features

-   Stability-aware training loop
-   JSON state serialization
-   LaTeX report generation
-   Modular class-based architecture
-   Compatible with hybrid quantum pipelines

------------------------------------------------------------------------

## Installation

Requires:

-   Python 3.9+
-   PyTorch

Optional: - PennyLane (for quantum circuit integration)

------------------------------------------------------------------------

## Basic Usage

``` python
from sarayut_hqcnn_v3_5_standalone import SarayutStabilityModel

def total_loss(theta):
    prediction = torch.sum(theta ** 2)
    sigma = torch.var(theta)
    loss = prediction + 0.75 * sigma
    return loss, sigma

model = SarayutStabilityModel(theta_init=[0.5, -0.3])
model.train(total_loss, steps=100)
model.save_state("model_state.json")
```

------------------------------------------------------------------------

## Output Artifacts

-   JSON model state
-   LaTeX table for IEEE publication
-   Training history logs

------------------------------------------------------------------------

## Research Notes

This framework provides a practical implementation of perturbation-based
stability regularization for hybrid quantum machine learning systems.

Further validation is required for: - Multi-qubit scaling analysis -
Hardware noise benchmarking - Statistical robustness evaluation

------------------------------------------------------------------------

## License

Research and educational use.
