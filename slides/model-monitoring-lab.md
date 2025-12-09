---
marp: true
theme: default
paginate: true
---

# Lab: Model Monitoring

**Objective**: Detect data drift in a simulated production environment.

## Task 1: Baseline
- Train a simple classifier on "Reference" data.
- Log performance metrics.

## Task 2: Simulation
- Simulate a stream of "Production" data that gradually changes statistics (e.g., add noise, shift mean).
- Run predictions.

## Task 3: Detection with Evidently
- Use `evidently` to compare Reference vs. Current batches.
- Generate a Data Drift Report (HTML).
- Identify which features drifted.
