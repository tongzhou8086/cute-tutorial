# 04 - Vector Add

This is the first real GPU kernel chapter. It intentionally avoids layouts,
tiling, shared memory, TMA, mbarriers, and pipeline structure.

The goal is only to show the smallest useful CuTe DSL workflow:

1. wrap PyTorch tensors as CuTe tensors,
2. compile a `@cute.jit` launch function,
3. launch a `@cute.kernel`,
4. use CUDA thread/block indices,
5. read and write global tensors with ordinary indexing.

## Running

```bash
python 04-vector-add/main.py
```

Expected output:

```text
vector_add verify: OK
n=1000000 max_abs_err=0.0
```

You can change the vector length:

```bash
python 04-vector-add/main.py --n 12345
```

## Kernel

The device kernel looks like a direct CUDA vector-add kernel:

```python
@cute.kernel
def vector_add_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    n = a.shape[0]

    if idx < n:
        c[idx] = a[idx] + b[idx]
```

The only CuTe-specific pieces here are:

- `cute.arch.thread_idx()`, `block_idx()`, and `block_dim()` replace CUDA's
  `threadIdx`, `blockIdx`, and `blockDim`.
- `a[idx]`, `b[idx]`, and `c[idx]` are ordinary CuTe tensor indexing operations.
- `if idx < n` is a dynamic device-side branch.

There is no `local_tile`, `group_modes`, `tma_partition`, `make_layout`, or
manual loop over tile elements in this chapter.

## Launch Wrapper

The `@cute.jit` function runs on the host during tracing. It computes the grid
shape and launches the kernel:

```python
@cute.jit
def vector_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    n = a.shape[0]
    grid = cute.ceil_div(n, THREADS_PER_BLOCK)
    vector_add_kernel(a, b, c).launch(
        grid=(grid, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
    )
```

This is the same separation we will use later:

- `@cute.jit` sets up launch-time structure.
- `@cute.kernel` is the GPU code.

## Host Code

On the Python side, PyTorch tensors are wrapped with DLPack and passed to
`cute.compile`:

```python
a_cute = from_dlpack(a, assumed_align=16)
b_cute = from_dlpack(b, assumed_align=16)
c_cute = from_dlpack(c, assumed_align=16)

compiled = cute.compile(vector_add, a_cute, b_cute, c_cute)
compiled(a_cute, b_cute, c_cute)
```

`cute.compile` traces once and returns an executor. Calling the executor launches
the compiled kernel.

### About `assumed_align`

For this scalar vector-add kernel, `assumed_align=16` is not essential. The code
does one scalar load from `a`, one scalar load from `b`, and one scalar store to
`c` per thread, so it does not rely on a 16-byte vectorized memory instruction.

In other words, if your CuTe DSL version accepts the default, this chapter would
still be conceptually correct as:

```python
a_cute = from_dlpack(a)
b_cute = from_dlpack(b)
c_cute = from_dlpack(c)
```

We keep `assumed_align=16` in the runnable code as a conservative pointer
alignment promise to the compiler. CUDA/PyTorch allocations normally satisfy at
least this alignment, and later chapters use alignment more directly for
vectorized copies, TMA descriptors, and layout-sensitive memory operations.

## Files

- `main.py` - minimal CuTe DSL vector-add kernel and host verification.
