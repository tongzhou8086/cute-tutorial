import argparse

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


THREADS_PER_BLOCK = 256


@cute.kernel
def vector_add_kernel(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    idx = bidx * bdim + tidx
    n = a.shape[0]

    if idx < n:
        c[idx] = a[idx] + b[idx]


@cute.jit
def vector_add(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    n = a.shape[0]
    grid = cute.ceil_div(n, THREADS_PER_BLOCK)
    vector_add_kernel(a, b, c).launch(
        grid=(grid, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal CuTe DSL vector add")
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    a = torch.randn(args.n, device="cuda", dtype=torch.float32)
    b = torch.randn(args.n, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    # `assumed_align=16` is not required for this scalar vector-add kernel.
    # It is a conservative compiler hint that becomes useful in later chapters
    # for vectorized memory operations and TMA.
    a_cute = from_dlpack(a, assumed_align=16)
    b_cute = from_dlpack(b, assumed_align=16)
    c_cute = from_dlpack(c, assumed_align=16)

    compiled = cute.compile(vector_add, a_cute, b_cute, c_cute)
    compiled(a_cute, b_cute, c_cute)
    torch.cuda.synchronize()

    expected = a + b
    torch.testing.assert_close(c, expected)
    max_abs_err = (c - expected).abs().max().item()
    print(f"vector_add verify: OK")
    print(f"n={args.n} max_abs_err={max_abs_err}")


if __name__ == "__main__":
    main()
