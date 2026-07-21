import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir


def dump_block(label: str) -> None:
    block = ir.InsertionPoint.current.block
    print(f"\n--- {label} ---")
    operations = list(block.operations)
    if not operations:
        print("<empty>")
        return
    for op in operations:
        print(op.get_asm(assume_verified=True))


@cute.jit
def choose_path(x: cutlass.Int32, use_fast: cutlass.Constexpr):
    dump_block("entry")

    if use_fast:
        print("[trace] building fast branch")
        cute.printf("[runtime] y = %d\n", x + cutlass.Int32(1))
    else:
        print("[trace] building slow branch")
        cute.printf("[runtime] y = %d\n", x + cutlass.Int32(100))
    dump_block("after plain if over constexpr")


if __name__ == "__main__":
    cute.compile(choose_path, cutlass.Int32(0), True)
