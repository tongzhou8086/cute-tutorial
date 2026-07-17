# 05 - 2D Tensor Copy

This chapter takes the vector-add kernel from `04-vector-add` and changes only
one thing: the tensors are now 2D.

There is still no tiling, shared memory, explicit layout construction, TMA, or
pipeline synchronization. The point is to make 2D tensor indexing feel ordinary
before introducing tile views.

## Running

```bash
python 05-2d-copy/main.py
```

Expected output:

```text
2d_copy verify: OK
shape=(256, 384) max_abs_err=0.0
```

## Kernel

The kernel still uses a 1D CUDA launch. Each thread owns one logical element.
The only new step is converting the flat thread index to `(row, col)`:

```python
linear_idx = bidx * bdim + tidx
rows, cols = src.shape
numel = rows * cols

if linear_idx < numel:
    row = linear_idx // cols
    col = linear_idx % cols
    dst[row, col] = src[row, col]
```

This is the same indexing arithmetic you would write in CUDA or Numba CUDA.
CuTe tensor indexing handles the tensor layout behind `src[row, col]`.

## Why This Chapter Exists

TMA examples operate on matrix tiles, so later chapters need the reader to be
comfortable with 2D tensor coordinates. This chapter introduces only that.

The next chapter keeps the same copy operation but changes the ownership model:
instead of each CTA receiving a flat range of elements, each CTA receives a
rectangular tile.

## Files

- `main.py` - minimal 2D CuTe DSL tensor copy and host verification.
