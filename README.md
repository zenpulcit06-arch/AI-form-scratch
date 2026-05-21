# AI From Scratch: Vectorized Polynomial & Logistic Regression (Fortran)

**Author:** Pulkit Jain — BITS Pilani |  Physics

> A low-level Machine Learning engine built in Modern Fortran — no PyTorch, no Scikit-Learn, just raw matrix calculus, gradient descent, and feature engineering.

---

## Overview

This repository implements a **Non-Linear Binary Classifier** — a single-neuron engine capable of learning non-linear decision boundaries using:

- **Logistic Regression** — probabilistic decision-making via Sigmoid activation
- **Polynomial Feature Mapping** — transforms multi-dimensional input into a full monomial feature space via combinatorial expansion
- **L2 Regularization (Ridge)** — prevents overfitting on high-degree polynomial models
- **Vectorized Matrix Operations** — the entire dataset is treated as a matrix $X$, with weights as a vector $w$

The engine moves beyond linear classification by combining these techniques into a clean, modular Fortran architecture.

---

## Repository Structure

```
AI-form-scratch/
├── linear_regression/
│   └── logistic_regression.f90   # Core math module: Fit, Standardize, Sigmoid, Accuracy
├── polynomial_regression/
│   └── polynomialreg.f90         # Main orchestrator & polynomial feature generator
├── makefile                      # Cross-platform build system
└── README.md
```

---

## Key Features

| Feature | Description |
|---|---|
| **Polynomial Expansion** | Transforms $n$ input features into all monomials up to degree $D$ — a $\binom{n+D}{D}-1$ dimensional feature space |
| **Combinatorial Feature Engine** | Enumerates all weak compositions via recursive `li_sol` — generates cross-terms like $x_1 x_2$, $x_1^2 x_2$, etc. |
| **Vectorized Predictions** | Computes $\hat{y} = \sigma(Xw + b)$ as a single matrix operation |
| **Z-Score Standardization** | Scales all features to mean $= 0$, S.D. $= 1$ to prevent gradient explosion |
| **L2 Regularization** | Adds a weight-decay penalty to the loss to combat overfitting |
| **Gradient Descent** | Optimized with Fortran's `do concurrent` for parallel-ready performance |
| **Modular Architecture** | Math engine (`logistic_regression`) and feature generator (`polynomialreg`) fully separated |

---

## Design Choices & Architecture

### 1. Why Fortran?

Python is standard for AI, but Fortran was chosen for its native high-performance array handling. Built-in functions like `matmul()`, `transpose()`, and `do concurrent` allow this engine to perform matrix operations at near-hardware speeds — a natural precursor to future GPU implementation.

### 2. Multivariate Matrix Logic (Vectorization)

Instead of looping through individual samples, the engine treats the entire dataset as a single matrix operation:

$$\hat{y} = \sigma(Xw + b)$$

This eliminates per-sample loops and enables significant computational speedups on large datasets.

### 3. Polynomial Feature Mapping via Combinatorics

The `Polynomial_reg` module generates all monomials up to degree $D$ over $n$ input features. For $n=2$, $D=2$:

$$x_1,\ x_2,\ x_1^2,\ x_1 x_2,\ x_2^2$$

The total number of expanded features is $\binom{n+D}{D} - 1$. This is computed using the log-gamma identity for numerical stability:

$$\binom{n+D}{D} = \text{nint}\!\left(\exp\!\left(\ln\Gamma(n+D+1) - \ln\Gamma(n+1) - \ln\Gamma(D+1)\right)\right)$$

The monomial exponent combinations are enumerated by `li_sol` — a recursive subroutine that generates all weak compositions of degree $d$ into $n$ non-negative parts. `sol_count` is passed by reference through the call stack rather than stored as module-level state, making the routine fully re-entrant and safe for future parallel use.

### 4. L2 Regularization (Ridge)

High-degree polynomials are prone to overfitting — they become too "jagged" trying to pass through every data point. L2 regularization adds a penalty term to both the loss and the gradient:

$$dw = \frac{1}{m} X^T(\hat{y} - y) + \frac{\lambda}{m}w$$

This **weight decay** forces the model to prefer smaller weights, producing smoother, more generalizable curves.

### 5. Z-Score Standardization

Polynomial terms like $x^{10}$ can be exponentially larger than $x^1$. Without scaling, the gradient explodes and training fails with `NaN`. The saved `mean` and `std` from training are also applied at prediction time so the input speaks the same language as the learned weights.

### 6. Sigmoid Activation

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Maps any real-valued output to a probability in $(0, 1)$, turning the regression into a binary classifier.

---

## How to Compile

Ensure `gfortran` is installed (via MinGW/MSYS2 on Windows, or natively on Linux/macOS).

The makefile auto-detects the OS and sets the correct binary extension and shell commands. It was written with assistance from Claude (Anthropic).

**Using Make:**
```bash
make
```

**Manual compilation:**
```bash
gfortran -O3 linear_regression/logistic_regression.f90 polynomial_regression/polynomialreg.f90 -o ai_engine
```

---

## Test Cases

### Test Case 1: The Parabola

The engine was verified on a 1D dataset where $y = 1$ at the extremes and $y = 0$ in the center — a non-linear boundary no linear model can learn.

| $x$ | Label |
|-----|-------|
| 1   | 0     |
| 5   | 1     |
| 9   | 0     |

**Result:** The model correctly suppressed the linear weight (Weight $\approx 0$) and assigned a strong positive weight to the $x^2$ term, confirming it learned the parabolic structure.

| Metric | Value |
|--------|-------|
| Initial Loss | $\approx 0.69$ (random baseline) |
| Final Loss | $\approx 0.08$ |

---

### Test Case 2: The Circle (`circle.csv`) — Phase 1 (1D, broken)

A 2D radial dataset with 50 points fed through the Phase 1 single-column engine. The model only saw $x_1$, making it structurally impossible to learn the boundary.

| Metric | Value |
|--------|-------|
| Final Loss | 0.3096 |
| MSE | 0.5019 |

**Observation:** MSE ~0.50 is essentially random-guess performance. This test exposed the Phase 1 limitation and directly motivated the multi-feature rewrite.

---

### Test Case 3: The Circle (`circle.csv`) — Phase 2 (2D, solved)

The same radial dataset, now with 200 samples and both features read correctly. The engine expanded $(x_1, x_2)$ into 5 features — $x_1,\ x_2,\ x_1^2,\ x_1 x_2,\ x_2^2$ — and trained on the full 2D structure.

**Run configuration:**

| Parameter | Value |
|-----------|-------|
| Dataset | `circle.csv` (200 samples, 2 features + label) |
| Degree | 2 |
| Learning Rate | 0.03 |
| Lambda (L2) | 0.005 |
| Iterations | 2,500 |

**Loss convergence:**

```
0.24396  →  0.18395  →  0.15638  →  0.13974  →  0.12832  →  0.12878 (final)
```

**Learned weights:**

| Feature | Term | Weight |
|---------|------|--------|
| $w_1$ | $x_1$ | 0.254 |
| $w_2$ | $x_2$ | −0.105 |
| $w_3$ | $x_1^2$ | **3.217** |
| $w_4$ | $x_1 x_2$ | 0.145 |
| $w_5$ | $x_2^2$ | **2.928** |
| Bias | — | 2.062 |

**Final metrics:**

| Metric | Value |
|--------|-------|
| Final Loss | 0.1288 |
| MSE | 0.0306 |

**Sample prediction:**

```
Input: x₁ = 2.0,  x₂ = 3.0
Probability:  1.000
Classification: POSITIVE (1)   ✓  (point is far outside the origin — correctly outer)
```

**Key insight:** The dominant weights are $w_3 \approx 3.22$ and $w_5 \approx 2.93$ — the $x_1^2$ and $x_2^2$ terms. The model independently discovered that the decision boundary is determined by the sum of squared coordinates, i.e. $x_1^2 + x_2^2 \approx \text{const}$. This is the correct mathematical form of a circle. The MSE dropped from 0.50 (Phase 1) to 0.031 (Phase 2) — a 16× improvement.

---

## Mistakes & Lessons Learned

<details>
<summary><strong>1. "Actual vs Formal" Argument Mismatch</strong></summary>

**Error:** The `main` program passed `lamda` to `subroutine fit`, but the subroutine interface wasn't updated to receive it — causing a silent mismatch.

**Fix:** Synchronized the module interface and used a clean-recompile strategy to ensure `.mod` files were fully regenerated.

</details>

<details>
<summary><strong>2. Scalar Penalty in the Gradient</strong></summary>

**Error:** Used `sum(weight)` in the L2 gradient calculation — this collapsed all weights into one scalar, destroying per-feature precision.

**Fix:** Corrected to use the full vector `weight` for an element-wise penalty.

</details>

<details>
<summary><strong>3. The <code>main</code> Conflict</strong></summary>

**Error:** `program main` was defined in both the module file and the application file, causing a *"multiple definition of main"* linker error.

**Fix:** Separated all math logic into pure modules; kept execution logic in a single `program` file.

</details>

<details>
<summary><strong>4. Memory Allocation Crashes</strong></summary>

**Error:** Attempted to access `x_poly` and `weight` arrays before the user had provided `degree` and `sample_size`.

**Fix:** All `allocate()` calls now happen after runtime input is collected.

</details>

<details>
<summary><strong>5. Standardization Not Applied at Prediction Time</strong></summary>

**Error:** The model predicted on raw input values, while the weights were trained on standardized features — a systematic mismatch.

**Fix:** The `mean` and `std` computed during training are saved and reapplied to every input at prediction time.

</details>

<details>
<summary><strong>6. Compilation Flag Errors</strong></summary>

**Error:** Used the `-I` flag pointing to `.f90` and `.exe` files directly.

**Fix:** `-I` is for directories containing compiled `.mod` files. Compiling the module and program together in a single command is cleaner.

</details>

<details>
<summary><strong>7. Single-Column Input Cannot Model 2D Boundaries</strong></summary>

**Error:** Fed only the first column of `circle.csv` into the Phase 1 engine — the model saw one coordinate per point but not both, making it impossible to learn a radial boundary.

**Insight:** The high MSE (~0.50) was not a tuning problem — it was a structural one. True 2D classification requires both $x_1$ and $x_2$ plus their cross-terms. This directly motivated Phase 2.

</details>

<details>
<summary><strong>8. Module-Level <code>sol_count</code> Shared State</strong></summary>

**Error:** `sol_count` was a module-level variable incremented inside `li_sol`. A second call to `make_poly` (e.g. at prediction time) would pick up the counter value left over from training instead of starting fresh, silently writing into the wrong rows of `res_matrix`.

**Fix:** Removed `sol_count` from module scope entirely. It is now a local variable inside `make_poly` and passed into `li_sol` as an `intent(inout)` argument. Each call to `make_poly` owns its own counter — no shared state, no `reset()` subroutine needed, safe for future parallel use.

</details>

<details>
<summary><strong>9. Re-allocation Crash in <code>standardize</code></strong></summary>

**Error:** `z_score` is declared `allocatable` in `standardize`. When called a second time (at prediction), Fortran raised a runtime error: *"Attempting to allocate already allocated variable"* — because the variable was still marked as allocated from the first call.

**Fix:** Added an `if (allocated(z_score)) deallocate(z_score)` guard immediately before the `allocate` call. Same fix applied to `y_predicted` in `fit` for consistency.

</details>

---

## Roadmap

- [x] Phase 1 — Polynomial Logistic Regression with L2 Regularization (1D input)
- [x] Phase 2 — Multi-feature input, combinatorial monomial expansion, concurrent-safe recursion
- [ ] Phase 3 — Multi-layer Neural Network (Hidden Layers)
- [ ] Phase 4 — Vectorized Backpropagation algorithm
- [ ] Phase 5 — Multi-class Classification (Softmax)
- [ ] Phase 6 — CUDA/GPU acceleration for large matrices
- [ ] Phase 7 — Link with BLAS/LAPACK for optimized matrix operations

---

## Part of the *AI From Scratch* Series

This project is one step in a larger journey to build AI primitives from first principles using low-level languages. Each module builds on the last — with full transparency into design decisions, mathematical derivations, and hard-won lessons from real bugs.