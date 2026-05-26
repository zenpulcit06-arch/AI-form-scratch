# AI From Scratch: Vectorized Polynomial & Logistic Regression + Neural Network (Fortran)

**Author:** Pulkit Jain — BITS Pilani | Physics

> A low-level Machine Learning engine built in Modern Fortran — no PyTorch, no Scikit-Learn, just raw matrix calculus, gradient descent, and feature engineering.

---

## Transparency Statement

This project is a **genuine learning exercise**. I am a physics student teaching myself the mathematics and implementation of machine learning from first principles. To be transparent about how this was built:

**Written entirely by me (Pulkit):**
- All Fortran source code — every line was typed by me, not generated
- All architectural decisions (module structure, derived types, subroutine design)
- The mathematical reasoning behind each design choice
- Debugging — I identified and reasoned through every runtime error myself

**Where Claude (Anthropic) assisted:**
- Socratic guidance — Claude asked questions rather than giving answers, helping me derive backpropagation, forward pass equations, and gradient shapes myself
- Pointed out bugs after I had already written the code (e.g. `matmul` vs element-wise confusion, vanishing gradient diagnosis)
- Wrote the makefile (both Phase 1 and Phase 3 versions) — I am not familiar with makefile syntax
- Suggested the ReLU fix when sigmoid activations caused the vanishing gradient problem
- Explained the `spread()` intrinsic for bias broadcasting
- Guided derivation of generalized backpropagation and He initialization for Phase 4

**What this means:** The understanding is mine. The derivations, the dimension reasoning, the architecture choices — I worked through all of these. Claude functioned as a teacher who refused to just give answers, not as a code generator.

---

## Overview

This repository implements a **Non-Linear Binary Classifier** — a neural engine capable of learning non-linear decision boundaries using two approaches:

**Phase 1–2: Polynomial Logistic Regression**
- Logistic Regression with Sigmoid activation
- Polynomial Feature Mapping via combinatorial monomial expansion
- L2 Regularization (Ridge) to prevent overfitting
- Vectorized Matrix Operations across the full dataset

**Phase 3: Neural Network (Single Hidden Layer)**
- One hidden layer with ReLU activation
- Sigmoid output neuron for binary classification
- Full backpropagation derived from the chain rule
- No manual feature engineering — the network learns its own features

**Phase 4: Deep Neural Network (Arbitrary Depth)**
- N hidden layers with ReLU activation
- Generalized backpropagation via delta recurrence
- He weight initialization to prevent vanishing gradients
- Clean modular architecture — `layer` derived type array replaces hardcoded weights

---

## Repository Structure

```
AI-from-scratch/
├── linear_regression/
│   └── logistic_regression.f90   # Core math module: Fit, Standardize, Sigmoid, Accuracy
├── polynomial_regression/
│   └── polynomialreg.f90         # Polynomial feature generator + legacy main program
├── neural_network/
│   └── neural_network.f90        # Phase 3 neural network module (single hidden layer)
├── network/
│   └── network.f90               # Phase 4 network module: arbitrary depth, generalized backprop
├── main.f90                      # Main program — supports N-layer network
├── main_phase3.f90               # Archived Phase 3 main program
├── makefile                      # Cross-platform build system (written with Claude assistance)
└── README.md
```

---

## Key Features

| Feature | Description |
|---|---|
| **Polynomial Expansion** | Transforms $n$ input features into all monomials up to degree $D$ |
| **Combinatorial Feature Engine** | Enumerates all weak compositions via recursive `li_sol` |
| **Arbitrary Depth Network** | N hidden layers, neuron counts decided at runtime |
| **Generalized Backpropagation** | Delta recurrence: $\delta^{[l]} = \delta^{[l+1]} W_{l+1}^T \odot \text{ReLU}'(H_l)$ |
| **He Initialization** | Scales weights by $\sqrt{2/n_{prev}}$ to prevent vanishing gradients in deep networks |
| **ReLU Activation** | Hidden layers use ReLU to prevent vanishing gradients |
| **Sigmoid Output** | Binary classification output with cross-entropy loss |
| **Vectorized Predictions** | Full dataset processed as single matrix operations |
| **Z-Score Standardization** | Scales features to mean $= 0$, S.D. $= 1$ (polynomial model) |
| **L2 Regularization** | Weight-decay penalty to combat overfitting (polynomial model) |
| **Modular Architecture** | Math engine, feature generator, and neural network fully separated |

---

## Design Choices & Architecture

### 1. Why Fortran?

Python is standard for AI, but Fortran was chosen for its native high-performance array handling. Built-in functions like `matmul()`, `transpose()`, `spread()`, and `do concurrent` allow this engine to perform matrix operations at near-hardware speeds.

### 2. Multivariate Matrix Logic (Vectorization)

Instead of looping through individual samples, the engine treats the entire dataset as a single matrix operation:

$$\hat{y} = \sigma(Xw + b)$$

### 3. Polynomial Feature Mapping via Combinatorics

The `Polynomial_reg` module generates all monomials up to degree $D$ over $n$ input features. The total number of expanded features is $\binom{n+D}{D} - 1$.

### 4. Phase 3 Neural Network Architecture

The `network` derived type stores all parameters:

```fortran
type :: network
    real(real64), allocatable :: W1(:,:), W2(:,:), b1(:), b2(:), H(:,:)
end type
```

**Forward pass:**
$$H = \text{ReLU}(XW_1 + b_1)$$
$$\hat{y} = \sigma(HW_2 + b_2)$$

**Backward pass (chain rule):**
$$dW_2 = \frac{1}{m} H^T(\hat{y} - y)$$
$$dW_1 = \frac{1}{m} X^T \left[(\hat{y} - y)W_2^T \odot \text{ReLU}'(H)\right]$$

### 5. Phase 4 Deep Network Architecture

The `layer` derived type stores per-layer parameters:

```fortran
type :: layer
    real(real64), allocatable :: W(:,:), b(:), H(:,:)
end type

type(layer), allocatable :: net(:)   ! array of N layers
```

**Forward pass (generalized):**

For $l = 1$ to $N-1$:
$$H_l = \text{ReLU}(H_{l-1} W_l + b_l), \quad H_0 = X$$

For $l = N$:
$$\hat{y} = \sigma(H_{N-1} W_N + b_N)$$

**Backward pass (delta recurrence):**

$$\delta^{[N]} = \hat{y} - y$$

For $l = N$ down to $1$:
$$dW_l = \frac{1}{m} H_{l-1}^T \cdot \delta^{[l]}$$
$$db_l = \frac{1}{m} \sum \delta^{[l]}$$
$$\delta^{[l-1]} = \delta^{[l]} W_l^T \odot \text{ReLU}'(H_{l-1})$$

### 6. He Weight Initialization

Initial implementation used fixed `0.01` scaling. This caused vanishing gradients in deeper networks — loss stuck at ~0.644 (random baseline). He initialization scales weights by the number of input neurons:

$$W \sim \text{Uniform}(-1, 1) \times \sqrt{\frac{2}{n_{prev}}}$$

This keeps the variance of activations stable across layers regardless of depth.

### 7. Why ReLU for Hidden Layers?

Sigmoid's derivative is at most 0.25, so gradients shrank to near-zero through multiple layers. ReLU's derivative is 1 for positive values, so gradients flow without shrinking.

### 8. Bias Broadcasting with `spread()`

Adding a bias vector of size `neurons` to a matrix of size `samples × neurons` requires broadcasting. Fortran's `spread()` intrinsic replicates the vector across rows:

```fortran
net%H = relu(matmul(X, net%W1) + spread(net%b1, 1, size(X, 1)))
```

---

## How to Compile

Ensure `gfortran` is installed (via MinGW/MSYS2 on Windows, or natively on Linux/macOS).

```bash
make
```

---

## Benchmarks

### Test Case 1 & 2: Parabola and Circle (Polynomial Model)

| Metric | Value |
|--------|-------|
| Final Loss (circle, degree 2) | 0.1288 |
| MSE | 0.0306 |

### Test Case 3: Circle — Phase 3 Neural Network

| Parameter | Value |
|-----------|-------|
| Hidden neurons | 8 (ReLU) |
| Learning Rate | 0.1 |
| Iterations | 5000 |

| Metric | Value |
|--------|-------|
| Final Loss | 0.1393 |

### Test Case 4: Circle — Phase 4 Deep Network

Same dataset, now with arbitrary depth.

| Architecture | Loss |
|---|---|
| 1 hidden layer (8) | 0.0990 |
| 2 hidden layers (8, 8) | 0.0409 |

**Key insight:** Depth genuinely helps. Two hidden layers cut the loss by more than half compared to Phase 3. He initialization was essential — without it the 2-layer network was stuck at 0.644 (random baseline).

---

### Final Benchmark Goal: Handwritten Digit & Alphabet Recognition

The ultimate target for this engine is recognizing handwritten digits (MNIST, 10 classes) and handwritten alphabets (26 classes). This will require:

- **Phase 5** — Multi-class classification via Softmax output
- **Phase 6** — CUDA/GPU acceleration for large image datasets
- **Phase 7** — BLAS/LAPACK integration for optimized matrix operations

MNIST has 60,000 training samples of 28×28 pixel images (784 features). Successfully classifying it from scratch in Fortran — no frameworks, no libraries — would be a genuine demonstration that this engine works on real-world data.

---

## Mistakes & Lessons Learned

### Phase 1–2 Mistakes

<details>
<summary><strong>1. "Actual vs Formal" Argument Mismatch</strong></summary>

**Error:** The `main` program passed `lamda` to `subroutine fit`, but the interface wasn't updated to receive it.

**Fix:** Synchronized the module interface and used a clean-recompile strategy.

</details>

<details>
<summary><strong>2. Scalar Penalty in the Gradient</strong></summary>

**Error:** Used `sum(weight)` in the L2 gradient — collapsed all weights into one scalar, destroying per-feature precision.

**Fix:** Corrected to use the full vector `weight` for an element-wise penalty.

</details>

<details>
<summary><strong>3. The `main` Conflict</strong></summary>

**Error:** `program main` defined in both the module file and application file — multiple definition linker error.

**Fix:** Separated all math logic into pure modules; kept execution logic in a single `program` file.

</details>

<details>
<summary><strong>4. Memory Allocation Crashes</strong></summary>

**Error:** Accessed arrays before runtime input was collected.

**Fix:** All `allocate()` calls now happen after input is collected.

</details>

<details>
<summary><strong>5. Standardization Not Applied at Prediction Time</strong></summary>

**Error:** Predicted on raw values while weights were trained on standardized features — a systematic mismatch.

**Fix:** Saved `mean` and `std` from training and reapplied at prediction time.

</details>

<details>
<summary><strong>6. Single-Column Input Cannot Model 2D Boundaries</strong></summary>

**Error:** Fed only the first column of `circle.csv` — structurally impossible to learn a radial boundary. MSE ~0.50 was not a tuning problem, it was structural.

**Insight:** This directly motivated the Phase 2 multi-feature rewrite.

</details>

<details>
<summary><strong>7. Module-Level `sol_count` Shared State</strong></summary>

**Error:** `sol_count` was module-level, causing wrong row writes on a second call to `make_poly`.

**Fix:** Made it a local variable inside `make_poly`, passed as `intent(inout)` into `li_sol`.

</details>

<details>
<summary><strong>8. Re-allocation Crash in `standardize`</strong></summary>

**Error:** `z_score` raised a runtime error on second call — already allocated.

**Fix:** Added `if (allocated(z_score)) deallocate(z_score)` guard before `allocate`.

</details>

### Phase 3 Mistakes

<details>
<summary><strong>9. `matmul` vs Element-wise in Backward Pass</strong></summary>

**Error:** Wrote `matmul(net%H, (1-net%H))` in `dW1` — tried to matrix-multiply a 200×4 matrix with itself, which is dimensionally invalid.

**Fix:** Changed to `net%H * (1.0_real64 - net%H)`.

</details>

<details>
<summary><strong>10. Bias Broadcasting — Rank Mismatch</strong></summary>

**Error:** Tried to add a 1D bias vector directly to a 2D matrix.

**Fix:** Used `spread(net%b1, 1, size(X, 1))` to replicate the bias vector across all sample rows.

</details>

<details>
<summary><strong>11. Ambiguous `fit` Reference</strong></summary>

**Error:** Both `Layer` and `Logistic_regression` export a `fit` subroutine — ambiguous reference error.

**Fix:** Removed `use Logistic_regression` from `main.f90` since it was not needed there.

</details>

<details>
<summary><strong>12. Vanishing Gradient — Sigmoid in Hidden Layer</strong></summary>

**Error:** Used sigmoid for hidden layer activation. Loss stuck at ~0.64.

**Fix:** Replaced hidden layer with ReLU. Loss dropped from 0.64 to 0.14.

</details>

### Phase 4 Mistakes

<details>
<summary><strong>13. Fixed Weight Scaling in Deep Networks</strong></summary>

**Error:** Used `0.01` fixed scaling for all layers. With 2+ hidden layers, loss stuck at 0.644 — vanishing gradients again, this time from poor initialization rather than wrong activation.

**Fix:** Replaced with He initialization: `W * sqrt(2.0 / prev_neurons)`. Loss dropped to 0.0409 with 2 hidden layers.

**Lesson:** Weight initialization matters as much as architecture. The same vanishing gradient problem that ReLU fixes at the activation level, He initialization fixes at the scale level.

</details>

<details>
<summary><strong>14. `do concurrent` with Allocatable Arrays</strong></summary>

**Error:** Attempted to allocate inside `do concurrent` — not permitted by most Fortran compilers.

**Fix:** Replaced with a regular `do` loop for allocation, or called the dedicated `allocate_delta` subroutine.

</details>

<details>
<summary><strong>15. `net(0)` Index Out of Bounds</strong></summary>

**Error:** General backward pass loop accessed `net(i-1)%H` when `i=1`, giving `net(0)` which doesn't exist.

**Fix:** Added `if (i .ne. 1)` branch — use `X` directly when at the first layer.

</details>

---

## Roadmap

- [x] Phase 1 — Polynomial Logistic Regression with L2 Regularization (1D input)
- [x] Phase 2 — Multi-feature input, combinatorial monomial expansion, concurrent-safe recursion
- [x] Phase 3 — Single hidden layer Neural Network with ReLU + Backpropagation
- [x] Phase 4 — Arbitrary depth Neural Network with He initialization + Generalized Backpropagation
- [ ] Phase 5 — Multi-class Classification (Softmax)
- [ ] Phase 6 — CUDA/GPU acceleration for large matrices
- [ ] Phase 7 — Link with BLAS/LAPACK for optimized matrix operations
- [ ] **Final Benchmark** — Handwritten digit recognition (MNIST) and alphabet recognition from scratch in Fortran

---

## Part of the *AI From Scratch* Series

This project is one step in a larger journey to build AI primitives from first principles using low-level languages. Each module builds on the last — with full transparency into design decisions, mathematical derivations, hard-won lessons from real bugs, and honest acknowledgment of where AI assistance was used.