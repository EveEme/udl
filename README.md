# Uncertainty Disentanglement with Laplace Approximations
## Introduction
Layer-wise Laplace approximations for epistemic and aleatoric uncertainty disentanglement in neural networks.

This codebase include:
- a training and evaluation loop for three methods (CE Baseline, Laplace Approximation, and Dual Laplace Approximation)
- support for CIFAR-10 ResNet variants, including Wide ResNets
- script to reproduce the results `(train.py)`
- plotting utilities to recreate the plots

## Setup
Create virtual environment for `udl` and then use one of the commands:
- `python -m pip install .`
- `python -m pip install -e '.[dev]'` 

## Dataset
The Cifar-10 dataset will be downloaded automatically.

The Cifar10H dataset can be downloaded from here https://zenodo.org/records/8115942

## Run
For laplace experiments to run, first the ce baseline checkpoints have to be created.
- Main script is train.py 

## Plotting
For plotting the results one can use the script plot_dual_laplace_experiments.py with the command
`python plot_dual_laplace_experiments.py cifar10`
