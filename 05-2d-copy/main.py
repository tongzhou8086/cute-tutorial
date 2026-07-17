import argparse

import torch

import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


THREADS_PER_BLOCK = 256


@cute.kernel
def copy_2d_kernel(src: cute.Tensor, dst: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    linear_idx = bidx * bdim + tidx
    rows, cols = src.shape
    numel = rows * cols

    if linear_idx < numel:
        row = linear_idx // cols
        col = linear_idx % cols
        dst[row, col] = src[row, col]


@cute.jit
def copy_2d(src: cute.Tensor, dst: cute.Tensor):
    rows, cols = src.shape
    grid = cute.ceil_div(rows * cols, THREADS_PER_BLOCK)
    copy_2d_kernel(src, dst).launch(
        grid=(grid, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal CuTe DSL 2D tensor copy")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=384)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    src = torch.randn((args.rows, args.cols), device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)

    src_cute = from_dlpack(src, assumed_align=16)
    dst_cute = from_dlpack(dst, assumed_align=16)

    compiled = cute.compile(copy_2d, src_cute, dst_cute)
    compiled(src_cute, dst_cute)
    torch.cuda.synchronize()

    torch.testing.assert_close(dst, src)
    max_abs_err = (dst - src).abs().max().item()
    print("2d_copy verify: OK")
    print(f"shape=({args.rows}, {args.cols}) max_abs_err={max_abs_err}")


if __name__ == "__main__":
    main()
