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
- Added `cpu_time` timer to `main.f90` to benchmark training performance

**Where Claude (Anthropic) assisted:**
- Socratic guidance — Claude asked questions rather than giving answers, helping me derive backpropagation, forward pass equations, gradient shapes, softmax, and binary file I/O myself
- Pointed out bugs after I had already written the code
- Wrote the makefile — I am not familiar with makefile syntax
- Suggested the ReLU fix when sigmoid activations caused the vanishing gradient problem
- Explained the `spread()` intrinsic for bias broadcasting
- Guided derivation of generalized backpropagation and He initialization for Phase 4
- Guided derivation of softmax and categorical cross-entropy for Phase 5
- Explained cache blocking and why BLAS is faster than naive matmul for Phase 6
- Guided the DGEMM parameter derivation for all forward and backward pass calls
- Guided Phase 7 binary file I/O — byte representation, base-256 arithmetic, bit manipulation, IDX format
- Assisted with makefile changes for OpenBLAS linking
- Guided Phase 8 mini-batch SGD — Fisher-Yates shuffle, epoch/iteration distinction, index-based batching
- Guided Phase 8 model saving — binary serialization of weights, int32/int64 type consistency

**What this means:** The understanding is mine. The derivations, the dimension reasoning, the architecture choices — I worked through all of these. Claude functioned as a teacher who refused to just give answers, not as a code generator.

---

## Overview

This repository implements a **Non-Linear Classifier** — a neural engine capable of learning non-linear decision boundaries using two approaches:

**Phase 1–2: Polynomial Logistic Regression**
- Logistic Regression with Sigmoid activation
- Polynomial Feature Mapping via combinatorial monomial expansion
- L2 Regularization (Ridge) to prevent overfitting
- Vectorized Matrix Operations across the full dataset

**Phase 3: Neural Network (Single Hidden Layer)**
- One hidden layer with ReLU activation
- Sigmoid output neuron for binary classification
- Full backpropagation derived from the chain rule

**Phase 4: Deep Neural Network (Arbitrary Depth)**
- N hidden layers with ReLU activation
- Generalized backpropagation via delta recurrence
- He weight initialization to prevent vanishing gradients
- Clean modular architecture — `layer` derived type array

**Phase 5: Multi-Class Classification**
- Softmax output layer replacing sigmoid
- Categorical cross-entropy loss
- One-hot encoding of integer labels
- Numerically stable softmax via max subtraction trick
- Backpropagation unchanged — $\delta^{[N]} = \hat{p} - y$ holds for softmax + cross-entropy

**Phase 6: BLAS/LAPACK Integration**
- All `matmul` calls replaced with DGEMM (OpenBLAS)
- Cache-blocked matrix multiplication — dramatically faster at MNIST scale
- Transpose operations handled inside DGEMM via `transa`/`transb` flags

**Phase 7: Large-Scale Data Pipeline**
- Binary file I/O from scratch — no external parsing libraries
- IDX format reader for MNIST (images and labels)
- Big-endian to little-endian byte swapping via bit manipulation
- Signed `integer(int8)` to `real(real64)` conversion with unsigned fix
- Pixel normalization to `[0.0, 1.0]`
- Separate train and test set evaluation
- L2 regularization added to deep network backward pass
- `resize_H` subroutine for forward-pass-only inference on arbitrary sample sizes

**Phase 8: Mini-Batch SGD + Model Saving**
- Mini-batch gradient descent replacing full-batch training
- Fisher-Yates shuffle for unbiased random batch ordering each epoch
- Index-based batching — shuffle an integer index array, never copy the data matrix
- Epoch/iteration distinction — training measured in epochs, not raw iterations
- Eliminated overfitting: test accuracy matches training accuracy
- New `sgd` module — clean separation from core network module
- Model saving and loading — binary serialization of weights and biases to disk
- New `SaveingData` module — `save_network` and `load_network` subroutines
- Classes derived from loaded network — no need to re-enter architecture on load
- **Target achieved: 98% test accuracy on MNIST**

---

## Repository Structure

```
AI-from-scratch/
├── linear_regression/
│   └── logistic_regression.f90   # Core math module: Fit, Standardize, Sigmoid, Accuracy
├── polynomial_regression/
│   └── polynomialreg.f90         # Polynomial feature generator
├── neural_network/
│   └── neural_network.f90        # Layer module (Phase 3) + Network module (Phase 4/5/6/7)
├── sgd/
│   └── sgd.f90                   # Mini-batch SGD module: sgd_fit, sgd_shuffle (Phase 8)
├── saving_file/
│   └── save_file.f90             # Model serialization: save_network, load_network (Phase 8)
├── file_system/
│   └── binfile.f90               # Binary file I/O module: bswap32, read_int32, read_images, read_labels
├── main.f90                      # Main program — MNIST pipeline (Phase 7/8)
├── main_phase3.f90               # Archived Phase 3 main program
├── main_phase4_5_6.f90           # Archived Phase 4/5/6 main program
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
| **He Initialization** | Scales weights by $\sqrt{2/n_{prev}}$ to prevent vanishing gradients |
| **ReLU Activation** | Hidden layers use ReLU to prevent vanishing gradients |
| **Softmax Output** | Multi-class classification with categorical cross-entropy loss |
| **Numerically Stable Softmax** | Subtracts $\max(z)$ before exponentiation to prevent overflow |
| **One-Hot Encoding** | Integer labels converted to one-hot vectors at runtime |
| **DGEMM via OpenBLAS** | Cache-blocked matrix multiply replacing all `matmul` calls |
| **Z-Score Standardization** | Scales features to mean $= 0$, S.D. $= 1$ (polynomial model) |
| **L2 Regularization** | Weight-decay penalty in both logistic regression and deep network |
| **Binary IDX Reader** | Reads raw MNIST byte files without any parsing library |
| **Big-Endian Byte Swap** | `bswap32` converts 4-byte big-endian integers via bit manipulation |
| **Pixel Normalization** | Converts `integer(int8)` pixels to `real(real64)` in `[0.0, 1.0]` |
| **Train/Test Evaluation** | Separate accuracy measurement on unseen test data |
| **resize_H** | Reallocates activation arrays for inference without resetting weights |
| **Mini-Batch SGD** | Stochastic gradient descent with configurable batch size |
| **Fisher-Yates Shuffle** | Unbiased random permutation of sample indices each epoch |
| **Index-Based Batching** | Shuffle integer indices only — no data matrix copying |
| **Model Saving** | Binary serialization of all layer weights and biases to disk |
| **Model Loading** | Restore full network from file — architecture inferred from saved dimensions |

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

**Forward pass:**
$$H = \text{ReLU}(XW_1 + b_1)$$
$$\hat{y} = \sigma(HW_2 + b_2)$$

**Backward pass (chain rule):**
$$dW_2 = \frac{1}{m} H^T(\hat{y} - y)$$
$$dW_1 = \frac{1}{m} X^T \left[(\hat{y} - y)W_2^T \odot \text{ReLU}'(H)\right]$$

### 5. Phase 4 Deep Network Architecture

**Forward pass (generalized):**

For $l = 1$ to $N-1$:
$$H_l = \text{ReLU}(H_{l-1} W_l + b_l), \quad H_0 = X$$

For $l = N$:
$$\hat{y} = \text{softmax}(H_{N-1} W_N + b_N)$$

**Backward pass (delta recurrence):**

$$\delta^{[N]} = \hat{y} - y$$

For $l = N$ down to $1$:
$$dW_l = \frac{1}{m} H_{l-1}^T \cdot \delta^{[l]} + \frac{\lambda}{m} W_l$$
$$db_l = \frac{1}{m} \sum \delta^{[l]}$$
$$\delta^{[l-1]} = \delta^{[l]} W_l^T \odot \text{ReLU}'(H_{l-1})$$

### 6. Phase 5 Multi-Class Classification

**Softmax** replaces sigmoid at the output layer:

$$p_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

**Categorical cross-entropy:**

$$L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_{ik} \log(\hat{p}_{ik})$$

**Key insight:** $\delta^{[N]} = \hat{p} - y$ — backpropagation is unchanged from binary case.

### 7. Phase 6 — Why DGEMM is Faster Than `matmul`

DGEMM from OpenBLAS uses cache blocking — computing small tiles that fit in CPU cache, maximizing data reuse before eviction. Setting `transa='T'` handles `transpose(X)` without extra memory allocation.

$$C = \alpha \cdot \text{op}(A) \times \text{op}(B) + \beta \cdot C$$

### 8. Phase 7 — Binary File I/O from Scratch

**Why binary?** A CSV of 60,000 MNIST images would store each pixel as human-readable text — `128` takes 3 bytes as text but 1 byte as binary. Binary is smaller and faster to read.

**Byte representation:** Every file is a sequence of bytes (0–255). Multi-byte integers use base-256 positional notation — the same as base-10 but with 256 as the base instead of 10.

**Big-endian vs little-endian:** MNIST stores integers most-significant-byte first (big-endian). Most modern CPUs are little-endian. `bswap32` corrects this via bit shifts:

```fortran
p1 = iand(ishft(n, -24), 255)          ! byte 1 → position 4
p2 = iand(ishft(n, -8), 65280)         ! byte 2 → position 3
p3 = ishft(iand(n, 65280), 8)          ! byte 3 → position 2
p4 = ishft(iand(n,   255), 24)         ! byte 4 → position 1
r  = ior(ior(ior(p1, p2), p3), p4)
```

**Signed byte fix:** Fortran's `integer(int8)` is signed (-128 to 127). Pixel value 200 is stored as -56. Fix: `if (val < 0) val = val + 256`.

**Stream access:** `open(..., form='unformatted', access='stream')` reads raw bytes sequentially. File position advances automatically after each `read`.

### 9. Phase 8 — Mini-Batch SGD

**Why mini-batch?** Full-batch gradient descent computes the exact gradient over all samples — smooth but slow to converge and prone to getting stuck. Mini-batch uses a random subset each step, introducing controlled noise that acts as implicit regularization.

**The epoch/iteration distinction:** With batch size $B$ and dataset size $m$, one epoch = $\lfloor m/B \rfloor$ weight updates. Training is now measured in epochs (full passes through the data), not raw iterations.

**Fisher-Yates shuffle:** At the start of each epoch, a random permutation of indices $1$ to $m$ is generated in $O(m)$ time with guaranteed uniformity — every permutation equally likely.

**Index-based batching:** Instead of physically shuffling the data matrices, only an integer index array is shuffled. Batch $k$ is then accessed as `X(idx(start:end), :)`. This avoids copying a $60000 \times 784$ matrix every epoch — a 3.4x speedup in practice.

**Effect on generalization:** Mini-batch noise prevents overfitting. Phase 7 full-batch achieved 100% training accuracy but only 62.5% test accuracy. Phase 8 mini-batch achieves 98% test accuracy with architecture `[512, 256, 128, 10]` — the generalization gap is eliminated.

### 10. Phase 8 — Model Saving

**File format:** Binary stream — same format used for MNIST. Compact and fast. The save file stores the full network state:

```
N (int64)                          — number of layers
for each layer:
    current_neurons (int64)        — size of b, cols of W
    prev_neurons (int64)           — rows of W
    W (real64 array)               — weight matrix
    b (real64 array)               — bias vector
```

**Architecture inference on load:** When loading, `current_neurons` and `prev_neurons` are read from the file — no need to re-enter the architecture. `classes` is derived as `size(net(N)%W, 2)` from the loaded network.

**Type consistency:** All integer metadata written as `int64` explicitly using `int(..., int64)` — mixing `int32` from `size()` with `int64` reads causes silent corruption (wrong values, not a crash).

---

## Limitations

### 1. No Validation Set
The engine currently evaluates on training accuracy and test accuracy only. There is no validation split for hyperparameter tuning. A proper pipeline would reserve samples from the training set as a validation set and never touch the test set until the final evaluation.

### 2. No Dropout
Dropout — randomly zeroing neurons during training — is one of the most effective regularization techniques for neural networks. It is not implemented.

### 3. No Adaptive Learning Rate
The learning rate is fixed throughout training. Methods like Adam, RMSProp, or even simple learning rate decay would allow faster early progress and finer convergence later.

### 4. Windows-Only GPU Path Blocked
Phase 9 (GPU acceleration via OpenACC) requires NVHPC, which does not support Windows natively. This requires WSL2 — still on the roadmap.

### 5. No Batch Normalization
Batch normalization stabilizes training in deep networks by normalizing layer inputs. Without it, deeper architectures (4+ layers) may train poorly even with He initialization.

### 6. Training Time at Scale
Architecture `[512, 256, 128, 10]` at 1000 epochs takes ~4.6 hours on CPU. GPU acceleration (Phase 9) is the next step to make deep experiments practical.

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
| Learning rate | 0.1 |
| Iterations | 5000 |
| Final Loss | 0.1393 |

### Test Case 4: Circle — Phase 4 Deep Network

| Architecture | Loss |
|---|---|
| 1 hidden layer (8) | 0.0990 |
| 2 hidden layers (8, 8) | 0.0409 |

### Test Case 5: Circle — Phase 5 Multi-Class

| Metric | Value |
|--------|-------|
| Final Loss | 0.0582 |
| Accuracy | 97.5% |

### Test Case 6: Circle — Phase 6 BLAS/LAPACK

| Metric | Value |
|--------|-------|
| Final Loss | ~0.066–0.069 |
| Accuracy | 98% |
| Training time (200 samples, 5000 iter) | 0.156 seconds |

### Test Case 7: MNIST — Phase 7 Binary Pipeline

Architecture `[128, 10]`, learning rate 0.1, lambda 0.001, 1000 iterations.

| Metric | Value |
|--------|-------|
| Final Training Loss | 0.00332 |
| Training Accuracy | 100% |
| Test Accuracy | 62.5% |
| Training Time | 25.8 seconds |

**Key insight:** Large generalization gap (100% train vs 62.5% test) indicates overfitting.

### Test Case 8: MNIST — Phase 8 Mini-Batch SGD

Architecture `[128, 10]`, learning rate 0.01, lambda 0.01, batch size 256.

| Epochs | Train Accuracy | Test Accuracy | Training Time |
|--------|---------------|---------------|---------------|
| 10 | 89.7% | 90.3% | 44s |
| 50 | 93.7% | 93.7% | 480s |

**Key insight:** Generalization gap eliminated. Phase 7 had a 37.5% gap (100% train vs 62.5% test). Phase 8 has a 0% gap — mini-batch noise acts as implicit regularization.

### Test Case 9: MNIST — Phase 8 Deep Architecture

Architecture `[512, 256, 128, 10]`, learning rate 0.01, lambda 0.01, batch size 256, 1000 epochs.

| Metric | Value |
|--------|-------|
| Final Loss | 6.02 × 10⁻⁴ |
| Training Accuracy | 100% |
| Test Accuracy | **98.0%** |
| Training Time | 16681s (~4.6 hours) |

**Key insight:** Target of >95% test accuracy achieved. Deeper architecture with more epochs converges to 98% — but training time highlights the need for GPU acceleration in Phase 9.

### Test Case 10: MNIST — Phase 8 Model Save/Load

Architecture `[128, 10]`, 5 epochs.

| Metric | Value |
|--------|-------|
| Test Accuracy (trained) | 87.9% |
| Test Accuracy (loaded from file) | 87.9% |

**Key insight:** Save and load reproduce identical results — weights are serialized and restored correctly.

---

## How to Compile

Ensure `gfortran` and OpenBLAS are installed (via MSYS2/UCRT64 on Windows).

```bash
make
```

**Installing OpenBLAS on Windows (MSYS2/UCRT64):**
```bash
pacman -S mingw-w64-ucrt-x86_64-openblas
```

**MNIST data files** (download and unzip):
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

Available from: https://github.com/fgnt/mnist

---

## Mistakes & Lessons Learned

### Phase 1–2 Mistakes

<details>
<summary><strong>1. "Actual vs Formal" Argument Mismatch</strong></summary>
Synchronized the module interface and used a clean-recompile strategy.
</details>

<details>
<summary><strong>2. Scalar Penalty in the Gradient</strong></summary>
Used `sum(weight)` in the L2 gradient — corrected to the full vector `weight` for element-wise penalty.
</details>

<details>
<summary><strong>3. The `main` Conflict</strong></summary>
Separated all math logic into pure modules; kept execution logic in a single `program` file.
</details>

<details>
<summary><strong>4. Memory Allocation Crashes</strong></summary>
All `allocate()` calls now happen after input is collected.
</details>

<details>
<summary><strong>5. Standardization Not Applied at Prediction Time</strong></summary>
Saved `mean` and `std` from training and reapplied at prediction time.
</details>

<details>
<summary><strong>6. Single-Column Input Cannot Model 2D Boundaries</strong></summary>
Directly motivated the Phase 2 multi-feature rewrite.
</details>

<details>
<summary><strong>7. Module-Level `sol_count` Shared State</strong></summary>
Made it a local variable inside `make_poly`, passed as `intent(inout)` into `li_sol`.
</details>

<details>
<summary><strong>8. Re-allocation Crash in `standardize`</strong></summary>
Added `if (allocated(z_score)) deallocate(z_score)` guard before `allocate`.
</details>

### Phase 3 Mistakes

<details>
<summary><strong>9. `matmul` vs Element-wise in Backward Pass</strong></summary>
Changed `matmul(net%H, (1-net%H))` to `net%H * (1.0_real64 - net%H)`.
</details>

<details>
<summary><strong>10. Bias Broadcasting — Rank Mismatch</strong></summary>
Used `spread(net%b1, 1, size(X, 1))` to replicate the bias vector across all sample rows.
</details>

<details>
<summary><strong>11. Ambiguous `fit` Reference</strong></summary>
Removed `use Logistic_regression` from `main.f90` since it was not needed there.
</details>

<details>
<summary><strong>12. Vanishing Gradient — Sigmoid in Hidden Layer</strong></summary>
Replaced hidden layer with ReLU. Loss dropped from 0.64 to 0.14.
</details>

### Phase 4 Mistakes

<details>
<summary><strong>13. Fixed Weight Scaling in Deep Networks</strong></summary>
Replaced with He initialization. Loss dropped to 0.0409 with 2 hidden layers.
</details>

<details>
<summary><strong>14. `do concurrent` with Allocatable Arrays</strong></summary>
Replaced with a regular `do` loop, then refactored into `allocate_delta`.
</details>

<details>
<summary><strong>15. `net(0)` Index Out of Bounds</strong></summary>
Added `if (i .ne. 1)` branch — use `X` directly when at the first layer.
</details>

### Phase 5 Mistakes

<details>
<summary><strong>16. `do concurrent` with Softmax</strong></summary>
Replaced with a regular `do` loop since softmax allocates internally.
</details>

<details>
<summary><strong>17. Integer Division in Accuracy</strong></summary>
Cast to real: `acc = real(right, real64)/real(sample_size, real64)`.
</details>

<details>
<summary><strong>18. `y` Allocated Before Layer Initialization</strong></summary>
Moved `allocate(y...)` to after all `intialize_layer` calls.
</details>

### Phase 6 Mistakes

<details>
<summary><strong>19. INTEGER(8)/INTEGER(4) Mismatch in DGEMM</strong></summary>
Wrapped all dimension arguments in every DGEMM call with `int()`.
</details>

<details>
<summary><strong>20. Missing Matrix Arguments in DGEMM Call</strong></summary>
Memorized the signature order: `transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc`.
</details>

### Phase 7 Mistakes

<details>
<summary><strong>21. `bswap32` — Mask and Shift Swapped</strong></summary>
For left-shift pieces, the correct order is `ishft(iand(n, mask), shift)` — isolate first, then shift.
</details>

<details>
<summary><strong>22. `integer(int8)` for Intermediate Byte Swap Values</strong></summary>
Used `int8` for `p1`–`p4` in `bswap32` — truncated bits above position 8. Fixed to `int32`.
</details>

<details>
<summary><strong>23. Reading Header Before Data</strong></summary>
Called `read_label` before reading the label file header. Added two `read_int32` calls to discard the label file header.
</details>

<details>
<summary><strong>24. `resize_H` Using `size(lay%H, 2)` After Deallocation</strong></summary>
Called `size(lay%H, 2)` after `deallocate(lay%H)` — undefined behavior. Fixed to `size(lay%W, 2)`.
</details>

<details>
<summary><strong>25. `10e-8` vs `1e-8` in Early Stopping</strong></summary>
`10e-8` = 10 × 10⁻⁸ = 10⁻⁷. Intended `1e-8`. Ten times too loose, causing premature early stopping.
</details>

<details>
<summary><strong>26. Early Stopping Incompatible with Full-Batch Gradient Descent</strong></summary>
With full-batch on 60,000 samples, loss decreases in very smooth tiny steps. Early stopping triggered after ~15,000 iterations instead of true convergence. Removed early stopping from `fit_network` entirely.
</details>

### Phase 8 Mistakes

<details>
<summary><strong>27. `bswap32` p2 Missing Left Shift</strong></summary>
`p2 = iand(ishft(n,-8), 255)` extracted byte 2 correctly but never shifted it left to position 3. Result: byte 2 landed at position 1 instead of position 3, giving wrong header values. Fixed: `p2 = iand(ishft(n,-8), 65280)`.
</details>

<details>
<summary><strong>28. `no_batch = 0` — Sample Size Was 96</strong></summary>
`bswap32` bug caused `n_size` to read as 96 instead of 60,000. Since 96 < 256 (batch size), integer division gave `no_batch = 0` and the batch loop never executed. Loss was NaN because `loss_accu / 0` is undefined.
</details>

<details>
<summary><strong>29. Learning Rate Too High for Mini-Batch</strong></summary>
Phase 7 used learning rate 0.1 with ~1000 weight updates total. Phase 8 with 10 epochs × 234 batches = 2340 updates at the same rate caused numerical explosion (NaN from epoch 1). Fixed by reducing learning rate to 0.01.
</details>

<details>
<summary><strong>30. Shuffling Full Data Matrix Each Epoch</strong></summary>
Original `sgd_shuffle` physically reordered all rows of `X` (60000 × 784) and `y` each epoch. Replaced with index-based shuffle — only a 60000-element integer array is shuffled. 3.4x speedup.
</details>

<details>
<summary><strong>31. `int32`/`int64` Mismatch in `save_network` — N Written as Wrong Type</strong></summary>
`write(10) size(net)` wrote `N` as `int32` (4 bytes) but `load_network` read it into an `int64` variable (8 bytes). The read consumed 8 bytes when only 4 were written, picking up garbage. Result: `N = 549755813890` instead of 2, causing an immediate allocation crash. Fixed: `write(10) int(size(net), int64)`.
</details>

<details>
<summary><strong>32. `int32`/`int64` Mismatch in `save_network` — Layer Dimensions Written as Wrong Type</strong></summary>
Same problem repeated for `size(net(i)%b)` and `size(net(i)%W,1)` — both returned `int32` by default. `load_network` read them as `int64`, producing garbage dimension values and an integer overflow crash during allocation. Fixed: `int(size(net(i)%b), int64)` and `int(size(net(i)%W,1), int64)`. Rule: always explicitly cast metadata to `int64` when writing binary files read back into `int64` variables.
</details>

---

## Roadmap

- [x] Phase 1 — Polynomial Logistic Regression with L2 Regularization (1D input)
- [x] Phase 2 — Multi-feature input, combinatorial monomial expansion
- [x] Phase 3 — Single hidden layer Neural Network with ReLU + Backpropagation
- [x] Phase 4 — Arbitrary depth Neural Network with He initialization
- [x] Phase 5 — Multi-class Classification (Softmax + Categorical Cross-Entropy)
- [x] Phase 6 — BLAS/LAPACK integration (OpenBLAS DGEMM)
- [x] Phase 7 — Binary data pipeline (IDX format, MNIST training and test evaluation)
- [x] Phase 8 — Mini-batch SGD + model saving (98% test accuracy achieved)
- [ ] Phase 9 — CUDA/GPU acceleration via WSL2 + NVHPC + OpenACC
- [ ] **Final Benchmark** — Sub-60s training at >98% test accuracy on GPU

---

## Part of the *AI From Scratch* Series

This project is one step in a larger journey to build AI primitives from first principles using low-level languages. Each module builds on the last — with full transparency into design decisions, mathematical derivations, hard-won lessons from real bugs, and honest acknowledgment of where AI assistance was used.