---
marp: true
theme: default
paginate: true
style: |
  section { background: white; font-family: 'Inter', sans-serif; font-size: 28px; padding-top: 0; justify-content: flex-start; }
  h1 { color: #1e293b; border-bottom: 3px solid #f59e0b; font-size: 1.6em; margin-bottom: 0.5em; margin-top: 0; }
  h2 { color: #334155; font-size: 1.2em; margin: 0.5em 0; }
  code { background: #f8f9fa; font-size: 0.85em; font-family: 'Fira Code', monospace; border: 1px solid #e2e8f0; }
  pre { background: #f8f9fa; border-radius: 6px; padding: 1em; margin: 0.5em 0; }
  pre code { background: transparent; color: #1e293b; font-size: 0.7em; line-height: 1.5; }
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
