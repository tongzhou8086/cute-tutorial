# 06 - CTA Tiles with `local_tile`

This chapter keeps the same matrix-copy operation from `05-2d-copy`, but
changes the ownership model:

- `05-2d-copy`: each CTA owns a flat range of elements.
- `06-local-tile-copy`: each CTA owns a rectangular tile.

This introduces `cute.local_tile` before any shared memory or TMA code appears.

For this copy alone, `05-2d-copy` is simpler and more natural. The point of
this chapter is not to improve the copy; it is to introduce the CTA-tile
ownership pattern used by GEMM, shared-memory staging, and TMA.

## Running

```bash
python 06-local-tile-copy/main.py
```

Expected output:

```text
local_tile_copy verify: OK
shape=(256, 384) tile=(16, 16) max_abs_err=0.0
```

This chapter avoids edge-tile predicates, so the matrix shape must be divisible
by the tile shape:

```bash
python 06-local-tile-copy/main.py --rows 512 --cols 512
```

## What `local_tile` Means

`local_tile` does not copy data. It creates a tensor view.

```python
src_tile = cute.local_tile(src, (TILE_M, TILE_N), (block_m, block_n))
dst_tile = cute.local_tile(dst, (TILE_M, TILE_N), (block_m, block_n))
```

For `TILE_M = TILE_N = 16`, CTA `(0, 0)` sees:

```text
src[0:16, 0:16]
```

CTA `(1, 0)` sees:

```text
src[16:32, 0:16]
```

CTA `(0, 1)` sees:

```text
src[0:16, 16:32]
```

The kernel then indexes within the CTA-local tile:

```python
row = tile_idx // TILE_N
col = tile_idx % TILE_N
dst_tile[row, col] = src_tile[row, col]
```

So `row` and `col` are local tile coordinates, not global matrix coordinates.
The tile view carries the global offset implied by `(block_m, block_n)`.

## Why Not Use the Chapter 5 Indexing?

You can. For a plain matrix copy, this is better:

```python
linear_idx = bidx * bdim + tidx
row = linear_idx // cols
col = linear_idx % cols
dst[row, col] = src[row, col]
```

That flat indexing style is ideal when each thread owns one independent global
element.

`local_tile` becomes useful when the CTA owns a whole rectangular work unit:

```python
src_tile = cute.local_tile(src, (TILE_M, TILE_N), (block_m, block_n))
dst_tile = cute.local_tile(dst, (TILE_M, TILE_N), (block_m, block_n))
```

Then the rest of the kernel can talk in tile-local coordinates:

```python
dst_tile[row, col] = src_tile[row, col]
```

That is the shape of the later kernels:

- a CTA owns an `A` tile and a `B` tile,
- the CTA stages those tiles through shared memory,
- TMA loads one CTA tile at a time,
- MMA consumes tile-shaped operands.

So chapter 6 is a stepping stone, not an optimization.

## Why This Matters for TMA

TMA loads and stores operate on tiles. Before introducing TMA descriptors,
`tma_partition`, mbarriers, or shared memory, we want the basic CuTe tile-view
operation to be unsurprising.

Later, the TMA chapter will still use `local_tile`; it will just replace the
manual per-thread copy in this chapter with a hardware TMA copy into shared
memory.

## Files

- `main.py` - CTA-tiled 2D copy using `cute.local_tile`.
