#!/usr/bin/env python3
"""
Knowledge Distillation Demo — Teacher (big MLP) → Student (small MLP)
Using MNIST digits for simplicity. Runs in ~60 seconds on CPU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import time
import os

np.random.seed(42)
torch.manual_seed(42)

# ── 1. Load Data (sklearn digits: 8x8 images, 10 classes) ──────────────
print("=" * 65)
print("Knowledge Distillation Demo")
print("Teacher: Big MLP (3 hidden layers, 512 neurons each)")
print("Student: Small MLP (1 hidden layer, 32 neurons)")
print("Dataset: sklearn digits (1797 samples, 64 features, 10 classes)")
print("=" * 65)

digits = load_digits()
X = torch.FloatTensor(digits.data / 16.0)  # normalize to [0, 1]
y = torch.LongTensor(digits.target)

# Full split for teacher training
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Teacher trains on ALL data
full_loader = DataLoader(TensorDataset(X_train_full, y_train_full), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

# Students train on LIMITED data (only 200 samples — simulates real-world scenario
# where you have a pretrained teacher but limited labeled data for your task)
X_train_small = X_train_full[:200]
y_train_small = y_train_full[:200]
small_loader = DataLoader(TensorDataset(X_train_small, y_train_small), batch_size=32, shuffle=True)

print(f"\nTeacher trains on: {len(X_train_full)} samples (full dataset)")
print(f"Students train on: {len(X_train_small)} samples (limited data)")
print(f"Test set:          {len(X_test)} samples")

# ── 2. Define Teacher (Big) and Student (Small) ────────────────────────
class Teacher(nn.Module):
    """Big model: 3 layers × 512 neurons."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        return self.net(x)

class Student(nn.Module):
    """Small model: 2 layers × 64 neurons."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 10),
        )
    def forward(self, x):
        return self.net(x)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds = model(X_b).argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)
    return correct / total

teacher = Teacher()
student_hard = Student()  # trained on hard labels (normal training)
student_distilled = Student()  # trained with distillation

print(f"\nTeacher params: {count_params(teacher):,}")
print(f"Student params: {count_params(student_hard):,}")
print(f"Size ratio:     {count_params(teacher) / count_params(student_hard):.0f}x smaller")

# ── 3. Train Teacher ───────────────────────────────────────────────────
print("\n--- Training Teacher (big model) ---")
optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001)
teacher.train()
for epoch in range(30):
    for X_b, y_b in full_loader:
        loss = F.cross_entropy(teacher(X_b), y_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

teacher_acc = evaluate(teacher)
print(f"Teacher accuracy: {teacher_acc:.1%}")

# ── 4. Train Student (Normal — Hard Labels) ───────────────────────────
print("\n--- Training Student (normal, hard labels only) ---")
optimizer = torch.optim.Adam(student_hard.parameters(), lr=0.001)
student_hard.train()
for epoch in range(50):
    for X_b, y_b in small_loader:
        loss = F.cross_entropy(student_hard(X_b), y_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

student_hard_acc = evaluate(student_hard)
print(f"Student (hard labels) accuracy: {student_hard_acc:.1%}")

# ── 5. Train Student (Distillation — Soft Labels from Teacher) ─────────
print("\n--- Training Student (distillation from teacher) ---")
optimizer = torch.optim.Adam(student_distilled.parameters(), lr=0.001)
temperature = 3.0  # soften the teacher's predictions
alpha = 0.5        # weight: 50% distillation loss, 50% hard label loss

teacher.eval()
student_distilled.train()
for epoch in range(50):
    for X_b, y_b in small_loader:
        # Teacher's soft predictions
        with torch.no_grad():
            teacher_logits = teacher(X_b)

        student_logits = student_distilled(X_b)

        # Soft loss: KL divergence between soft predictions
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction='batchmean'
        ) * (temperature ** 2)

        # Hard loss: normal cross-entropy
        hard_loss = F.cross_entropy(student_logits, y_b)

        # Combined loss
        loss = alpha * soft_loss + (1 - alpha) * hard_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

student_distilled_acc = evaluate(student_distilled)
print(f"Student (distilled) accuracy: {student_distilled_acc:.1%}")

# ── 6. Compare Everything ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

# Save models to compare sizes
torch.save(teacher.state_dict(), "/tmp/teacher.pth")
torch.save(student_hard.state_dict(), "/tmp/student_hard.pth")
torch.save(student_distilled.state_dict(), "/tmp/student_distilled.pth")

t_size = os.path.getsize("/tmp/teacher.pth") / 1024
sh_size = os.path.getsize("/tmp/student_hard.pth") / 1024
sd_size = os.path.getsize("/tmp/student_distilled.pth") / 1024

# Benchmark speed
x_bench = torch.randn(100, 64)
for m in [teacher, student_hard, student_distilled]:
    m.eval()
    for _ in range(50):
        with torch.no_grad(): m(x_bench)

def time_model(m, x, n=500):
    start = time.perf_counter()
    for _ in range(n):
        with torch.no_grad(): m(x)
    return (time.perf_counter() - start) / n * 1000

t_time = time_model(teacher, x_bench)
sh_time = time_model(student_hard, x_bench)
sd_time = time_model(student_distilled, x_bench)

print(f"\n{'Model':<30} {'Params':>10} {'Size':>10} {'Accuracy':>10} {'Speed':>12}")
print("-" * 75)
print(f"{'Teacher (big MLP)':<30} {count_params(teacher):>10,} {t_size:>8.1f}KB {teacher_acc:>9.1%} {t_time:>10.2f}ms")
print(f"{'Student (hard labels)':<30} {count_params(student_hard):>10,} {sh_size:>8.1f}KB {student_hard_acc:>9.1%} {sh_time:>10.2f}ms")
print(f"{'Student (distilled)':<30} {count_params(student_distilled):>10,} {sd_size:>8.1f}KB {student_distilled_acc:>9.1%} {sd_time:>10.2f}ms")
print("-" * 75)

improvement = student_distilled_acc - student_hard_acc
size_ratio = count_params(teacher) / count_params(student_hard)
speed_ratio = t_time / sd_time

print(f"\n{'KEY TAKEAWAYS':=^75}")
print(f"\n1. Student is {size_ratio:.0f}x SMALLER than teacher ({t_size:.0f}KB → {sd_size:.0f}KB)")
print(f"2. Student is {speed_ratio:.0f}x FASTER than teacher ({t_time:.2f}ms → {sd_time:.2f}ms)")
print(f"3. Student retains {student_distilled_acc/teacher_acc:.0%} of teacher's accuracy")
print(f"4. Distillation vs hard labels: {improvement:+.1%} on this dataset")
print(f"\nOn simple datasets, hard labels may work as well as distillation.")
print(f"Distillation shines on COMPLEX tasks with LIMITED labeled data —")
print(f"e.g., BERT (440MB) → DistilBERT (60MB) retains 97% accuracy at 7x smaller.")
print(f"\nThe point: you CAN deploy a tiny model that's nearly as good as the big one!")
