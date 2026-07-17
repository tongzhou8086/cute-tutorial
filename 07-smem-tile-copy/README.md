# 07 - SMEM Tile Copy

This chapter keeps the `local_tile` ownership model from
`06-local-tile-copy`, but inserts shared memory between the source and
destination:

```text
GMEM tile -> SMEM tile -> GMEM tile
```

There is still no TMA, no `group_modes`, no `tma_partition`, and no mbarrier.
The only new concepts are:

- `cute.make_layout` to describe the shared-memory tile shape and stride.
- `SmemAllocator` to allocate a shared-memory tensor.
- `cute.arch.barrier()` to make the SMEM writes visible before reads.

## Running

```bash
python 07-smem-tile-copy/main.py
```

Expected output:

```text
smem_tile_copy verify: OK
shape=(256, 384) tile=(32, 32) max_abs_err=0.0
```

## Shared-Memory Layout

The shared tile is explicitly row-major:

```python
smem_layout = cute.make_layout((TILE_M, TILE_N), stride=(TILE_N, 1))
```

For `TILE_M = TILE_N = 32`, this means:

- moving one row down advances by `32` elements,
- moving one column right advances by `1` element.

The allocation uses that layout:

```python
smem = SmemAllocator()
smem_tile = smem.allocate_tensor(
    cutlass.Float32, smem_layout, byte_alignment=1024
)
```

## Copy Flow

Each CTA owns one global tile:

```python
src_tile = cute.local_tile(src, (TILE_M, TILE_N), (block_m, block_n))
dst_tile = cute.local_tile(dst, (TILE_M, TILE_N), (block_m, block_n))
```

With `32 x 32` tiles and `256` threads per CTA, each CTA has `1024` tile
elements. The `while` loops therefore let each thread handle multiple elements
inside its CTA tile.

Then threads cooperatively copy the tile into shared memory:

```python
smem_tile[row, col] = src_tile[row, col]
```

After a CTA barrier, they copy from shared memory to the destination tile:

```python
dst_tile[row, col] = smem_tile[row, col]
```

This chapter is intentionally inefficient. Its job is to make SMEM allocation
and layout explicit before TMA replaces the first copy step.

## Files

- `main.py` - CTA-tiled copy through shared memory.
