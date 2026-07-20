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

    if cutlass.const_expr(use_fast):
        print("[trace] selected fast path")
        y = x + cutlass.Int32(1)
    else:
        print("[trace] selected slow path")
        y = x + cutlass.Int32(100)
    dump_block("after constexpr if")

    cute.printf("[runtime] y = {}\n", y)
    dump_block("after cute.printf")


if __name__ == "__main__":
    cute.compile(choose_path, cutlass.Int32(0), True)
