# ECG Time-Series Anomaly Detection using an LSTM Autoencoder

Author: Pawan Chaudhary

## Overview

This repository implements ECG time-series anomaly detection using a reconstruction-based
approach with an LSTM autoencoder in PyTorch. The objective is to model normal ECG
dynamics and identify anomalous heartbeats as sequences that the model fails to
reconstruct accurately.

Instead of framing the task as multi-class classification, the pipeline treats anomaly
detection as an open-world problem. The model is trained exclusively on normal ECG
sequences, and reconstruction error is used as a continuous anomaly score. A
percentile-based threshold converts this score into a binary anomaly decision.

The implementation is designed as a reproducible, CPU-friendly baseline that emphasizes
methodological correctness over architectural complexity.

## Problem Motivation

In real-world anomaly detection settings, abnormal patterns are often diverse, evolving,
and poorly labeled. Normal behavior, however, is typically abundant and consistent enough
to model reliably. Autoencoders are well suited to this setting because they learn a compact
representation of normal signals and reconstruct them accurately, while unfamiliar patterns
produce higher reconstruction error.

For ECG data, temporal structure is essential. The ordering of values across time carries
information about cardiac rhythm and morphology. Sequence models such as LSTMs can
capture these temporal dependencies while remaining relatively lightweight.

## Design Philosophy

This project is written with CPU-only execution in mind. Model size, batch configuration,
and preprocessing steps are chosen to ensure reasonable training time and memory usage
on a standard laptop environment.

The goal is not to maximize performance, but to establish a correct and interpretable
baseline that can be extended or scaled if additional computational resources become
available.

## Dataset

The ECG5000 dataset is used in this project. Each sample represents a fixed-length ECG
heartbeat sequence. Multiple heartbeat classes are provided, but only normal beats are
used for model training.

The dataset is publicly available and included in this repository for reproducibility.

## Method Summary

The workflow follows these steps:

- Load and preprocess ECG time-series data from .arff files
- Inspect class distribution and signal structure
- Split data into training, validation, and test sets without leakage
- Train an LSTM autoencoder using normal sequences only
- Use reconstruction loss as an anomaly score
- Select an anomaly threshold based on a percentile of training losses
- Evaluate performance on unseen normal and anomalous sequences
- Visually inspect original versus reconstructed signals

## Limitations

Reconstruction-based anomaly detection relies on threshold selection, which is sensitive
to distribution shifts in normal data. Model capacity also plays a critical role: models that
are too small may underfit normal patterns, while models that are too large may
reconstruct anomalous signals too well.

These limitations are acknowledged explicitly and motivate future extensions.

## References

This implementation is inspired by the following resources:

- https://www.youtube.com/watch?v=qN3n0TM4Jno
- https://github.com/curiousily/Getting-Things-Done-with-Pytorch/blob/master/06.time-series-anomaly-detection-ecg.ipynb

This repository represents a baseline implementation for learning and research purposes
and is not intended as a novel algorithmic contribution.
