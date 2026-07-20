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
def add_dynamicexpr(b: cutlass.Float32):
    dump_block("entry")

    a: cutlass.Float32 = 2.0
    dump_block("after a: cutlass.Float32 = 2.0")

    result = a + b
    dump_block("after result = a + b")

    print("[meta-stage] result =", result)
    dump_block("after Python print")

    cute.printf("[object-stage] result = %f\n", result)
    dump_block("after cute.printf")


if __name__ == "__main__":
    cute.compile(add_dynamicexpr, cutlass.Float32(0.0))
