# CuTe DSL Tutorial

This repository is a step-by-step tutorial for learning NVIDIA CuTe DSL from
small runnable kernels toward a Blackwell GEMM rewrite.

The long-term goal is to port the highly pipelined CUDA GEMM kernels in
`11-gemm` to CuTe DSL while preserving the same pipeline structure:

- CTA-tiled work decomposition,
- shared-memory staging,
- TMA global-to-shared loads,
- mbarrier synchronization,
- warp-specialized producer/consumer roles,
- Blackwell `tcgen05` MMA and TMEM epilogues.

The tutorial is organized into three modules.

## Module 1 - Core Foundations

Goal: understand how Python code becomes GPU code, and how CuTe describes
memory through layouts and tensors.

| Chapter | Topic | Introduces |
|---|---|---|
| `01-jit` | JIT tracing and `cute.compile` | Meta-stage vs object-stage execution; direct `@cute.jit` calls vs explicit compile. |
| `02-constexpr` | Static vs dynamic arguments | `cutlass.Constexpr`, dynamic runtime arguments, and how they appear in generated IR. |
| `03-tensor-layout` | Tensor layout | `from_dlpack`, `(Shape):(Stride)`, dynamic tensor layouts, static packed layouts, and CuTe layout printing. |

This module is the compiler/layout foundation. It is where we build the mental
model for CuTe's tensor abstraction before writing more serious kernels.

## Module 2 - SIMT Programming Model

Goal: use ordinary CUDA-style threads to move data and perform scalar work.

| Chapter | Topic | Introduces |
|---|---|---|
| `04-vector-add` | First GPU kernel | `@cute.kernel`, CUDA thread/block indices, 1D tensor indexing, host verification. |
| `05-2d-copy` | 2D tensor indexing | Mapping a flat thread id to `(row, col)` and copying `dst[row, col] = src[row, col]`. |
| `06-local-tile-copy` | CTA tile views | `cute.local_tile`; a CTA owns one rectangular tile and indexes within tile-local coordinates. |
| `07-smem-tile-copy` | Shared-memory tile staging | `cute.make_layout`, `SmemAllocator`, SMEM tensors, and CTA barriers. |

This module is intentionally CUDA-like. The kernels are not trying to be fast;
they introduce the control-flow and memory concepts that later hardware-level
kernels depend on.

## Module 3 - Hardware GEMM and Async Pipelines

Goal: learn the Hopper/Blackwell mechanisms used by high-performance GEMM
kernels: TMA, MMA/WGMMA, mbarriers, and overlapped pipelines.

| Chapter | Topic | Introduces |
|---|---|---|
| `08-group-modes` | Grouping tile modes | `local_tile(..., (None, None))` and `cute.group_modes` as pure tensor-view transforms. |
| `09-tma-atom-setup` | TMA descriptor setup | `cpasync.make_tiled_tma_atom` as the CuTe counterpart to host-side `CUtensorMap` setup. |
| `10-tma-load` | Minimal TMA load | `tma_partition`, `cute.copy(..., tma_bar_ptr=...)`, and mbarrier wait for one GMEM-to-SMEM tile load. |
| `11-gemm` | CUDA GEMM reference | The production CUDA/PTX kernels that later CuTe chapters will reimplement. |

This module is where the tutorial starts moving from scalar SIMT programming
toward the FlashInfer-style performance stack: TMA for memory movement, Tensor
Core MMA for compute, and asynchronous barriers/pipelines for overlap.

## Learning Path

The early chapters avoid advanced CuTe layout algebra. For example,
`05-2d-copy` is the preferred way to write a plain matrix copy. `06-local-tile-copy`
then rewrites the same copy using tile views, not because it is simpler, but
because later GEMM and TMA kernels naturally assign work by CTA tile.

The TMA-related concepts are split across several chapters:

1. `06-local-tile-copy` teaches tile ownership.
2. `07-smem-tile-copy` adds shared-memory tile storage.
3. `08-group-modes` explains why a tile may be represented as one grouped mode.
4. `09-tma-atom-setup` creates the TMA atom/descriptor objects.
5. `10-tma-load` combines those pieces to issue one TMA load.

This keeps `10-tma-load` from being the first place readers see `local_tile`,
`make_layout`, `group_modes`, shared memory, and mbarriers all at once.

## Running

Each chapter is self-contained:

```bash
python 04-vector-add/main.py
python 05-2d-copy/main.py
python 06-local-tile-copy/main.py
```

GPU chapters require a CUDA-capable NVIDIA GPU and a CuTe DSL installation.
The later Blackwell chapters require hardware and tooling that support SM100
features.

On this development machine, examples are typically run through SLURM:

```bash
srun --partition=dedicated --gres=gpu:nvidia_b200:1 python 07-smem-tile-copy/main.py
```

## Generated Files

CuTe DSL may emit IR, PTX, or cubin artifacts when dump/keep environment
variables are enabled. The repository ignores these generated files:

- `*.mlir`
- `*.ptx`
- `*.cubin`
- `__pycache__/`

## Status

Chapters `01` through `07` are the current introductory path.
Chapters `08` through `11` are being developed as the bridge toward TMA and the
CUDA GEMM reference.
