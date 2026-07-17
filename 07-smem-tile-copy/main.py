import argparse

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator


TILE_M = 32
TILE_N = 32
THREADS_PER_CTA = 256


@cute.kernel
def smem_tile_copy_kernel(src: cute.Tensor, dst: cute.Tensor, smem_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    block_m, block_n, _ = cute.arch.block_idx()
    block_dim, _, _ = cute.arch.block_dim()

    src_tile = cute.local_tile(src, (TILE_M, TILE_N), (block_m, block_n))
    dst_tile = cute.local_tile(dst, (TILE_M, TILE_N), (block_m, block_n))

    smem = SmemAllocator()
    smem_tile = smem.allocate_tensor(
        cutlass.Float32, smem_layout, byte_alignment=1024
    )

    tile_idx = tidx
    tile_elems = TILE_M * TILE_N
    while tile_idx < tile_elems:
        row = tile_idx // TILE_N
        col = tile_idx % TILE_N
        smem_tile[row, col] = src_tile[row, col]
        tile_idx += block_dim

    cute.arch.barrier()

    tile_idx = tidx
    while tile_idx < tile_elems:
        row = tile_idx // TILE_N
        col = tile_idx % TILE_N
        dst_tile[row, col] = smem_tile[row, col]
        tile_idx += block_dim


@cute.jit
def smem_tile_copy(src: cute.Tensor, dst: cute.Tensor):
    rows, cols = src.shape
    grid_m = cute.ceil_div(rows, TILE_M)
    grid_n = cute.ceil_div(cols, TILE_N)
    smem_layout = cute.make_layout((TILE_M, TILE_N), stride=(TILE_N, 1))

    smem_tile_copy_kernel(src, dst, smem_layout).launch(
        grid=(grid_m, grid_n, 1),
        block=(THREADS_PER_CTA, 1, 1),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="CuTe DSL SMEM tile copy")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rows % TILE_M != 0 or args.cols % TILE_N != 0:
        raise ValueError(
            f"--rows must be divisible by {TILE_M} and --cols by {TILE_N}; "
            "this chapter avoids edge-tile predicates."
        )

    torch.manual_seed(args.seed)
    src = torch.randn((args.rows, args.cols), device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)

    src_cute = from_dlpack(src, assumed_align=16)
    dst_cute = from_dlpack(dst, assumed_align=16)

    compiled = cute.compile(smem_tile_copy, src_cute, dst_cute)
    compiled(src_cute, dst_cute)
    torch.cuda.synchronize()

    torch.testing.assert_close(dst, src)
    max_abs_err = (dst - src).abs().max().item()
    print("smem_tile_copy verify: OK")
    print(
        f"shape=({args.rows}, {args.cols}) tile=({TILE_M}, {TILE_N}) "
        f"max_abs_err={max_abs_err}"
    )


if __name__ == "__main__":
    main()
