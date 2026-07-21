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
def choose_runtime(x: cutlass.Int32):
    dump_block("entry")

    if x > cutlass.Int32(0):
        print("[trace] building then branch")
        cute.printf("[runtime] positive\n")
    else:
        print("[trace] building else branch")
        cute.printf("[runtime] non-positive\n")
    dump_block("after dynamic if")


if __name__ == "__main__":
    cute.compile(choose_runtime, cutlass.Int32(0))
