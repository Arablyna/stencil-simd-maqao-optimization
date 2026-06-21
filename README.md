# Stencil SIMD Optimization with MAQAO

This repository contains a C performance optimization project focused on a 2D stencil computation over a complex-valued field.

The objective is to optimize a baseline scientific computing code using memory-layout transformations, loop restructuring, aligned memory allocation, SIMD vectorization and MAQAO performance analysis.

The final optimized version reduces the execution time from approximately **206 seconds** to **6.77 seconds** for `N = 5000` and `steps = 10`, achieving a speedup of about **30x**.

## Project Overview

The program simulates the time evolution of a 2D complex field:

```text
u(i, j) = ure(i, j) + i * uim(i, j)
```

At each time step, the update uses a 5-point stencil based on the discrete Laplacian:

```text
Δu(i, j) = u(i-1, j) + u(i+1, j) + u(i, j-1) + u(i, j+1) - 4u(i, j)
```

An energy value is also computed at each time step.

The initial implementation was dominated by expensive ASCII I/O, an Array of Structures memory layout, and multiple memory passes. The optimized implementation removes these bottlenecks and improves data locality and vectorization.

## Repository Structure

```text
.
├── README.md
├── src/
│   └── tp_optimized.c
└── report/
    └── tp_aoc_1.pdf
```

## Files

| File | Description |
|---|---|
| [`src/tp_optimized.c`](./src/tp_optimized.c) | Optimized C implementation of the stencil computation |
| [`report/tp_aoc_1.pdf`](./report/tp_aoc_1.pdf) | Full performance analysis report with MAQAO results |

## Optimization Strategy

The optimization process was incremental. Each step targeted a specific bottleneck.

| Step | Optimization | Purpose |
|---|---|---|
| 1 | AoS to SoA transformation | Separate real and imaginary parts to improve SIMD and cache behavior |
| 2 | Removal of ASCII I/O | Generate input directly in memory instead of using `fprintf` / `fscanf` |
| 3 | Loop fusion | Compute the energy during the update loop to avoid a second full memory pass |
| 4 | Stride-1 indexing | Improve memory locality by using contiguous row-major access |
| 5 | Aligned allocation | Use aligned memory to improve vector load/store efficiency |
| 6 | `restrict` pointers and OpenMP SIMD | Help the compiler vectorize the inner loop more aggressively |

## Performance Results

The following measurements were obtained for `N = 5000` and `steps = 10`.

| Version | Optimization Added | Runtime |
|---|---|---:|
| Baseline | Initial code | 206.00 s |
| optim1 | AoS to SoA | 200.00 s |
| optim2 | Remove ASCII I/O | 30.76 s |
| optim3 | Fuse energy computation | 29.28 s |
| optim4 | Stride-1 indexing | 29.18 s |
| optim5 | Aligned memory | 10.05 s |
| optim6 | `restrict` + SIMD | 6.77 s |

Final speedup:

```text
206.00 / 6.77 ≈ 30.4x
```

## Technical Details

The optimized version uses:

- Structure of Arrays memory layout
- direct in-memory input generation
- aligned allocation with `posix_memalign`
- compiler alignment hints with `__builtin_assume_aligned`
- `restrict` pointers to reduce aliasing uncertainty
- stride-1 memory access in the inner loop
- OpenMP SIMD directive with reduction
- loop fusion to reduce memory traffic
- MAQAO OneView for performance analysis

## Build

Compile the optimized version with GCC:

```bash
gcc -O3 -march=native -g -fno-omit-frame-pointer -fopenmp \
    -o app_opt src/tp_optimized.c -lm
```

## Run

Example run:

```bash
./app_opt 5000 10
```

Expected output format:

```text
Generating random input in memory
Computing 10 time steps
step 0 energy = ...
step 1 energy = ...
...
```

## MAQAO Analysis

The report includes MAQAO OneView analysis for both the baseline and optimized versions.

The analysis focuses on:

- runtime reduction
- memory access efficiency
- compiler optimization quality
- IPC and stalls
- SIMD vectorization opportunities
- cache and memory hierarchy behavior

## Skills Demonstrated

- C performance optimization
- scientific computing kernels
- 2D stencil computation
- memory-layout optimization
- SIMD-oriented programming
- OpenMP SIMD
- aligned memory allocation
- MAQAO profiling
- benchmark methodology
- performance analysis and technical reporting

## Relevance

This project demonstrates practical low-level performance optimization skills for CPU-based scientific computing workloads.

It is relevant for roles involving:

- High Performance Computing
- performance engineering
- scientific software development
- compiler-aware optimization
- numerical simulation
- CPU profiling and vectorization

## Author

**Arab Lina**  
M2 High Performance Computing  
Université Paris-Saclay / ENS Paris-Saclay
