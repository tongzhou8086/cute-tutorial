import cutlass
import cutlass.cute as cute


@cute.jit
def add_dynamicexpr(b: cutlass.Float32):
    a = cutlass.Float32(2.0)
    result = a + b
    print("[meta-stage] result =", result)              # host, at trace time
    cute.printf("[object-stage] result = %f\n", result)  # GPU, at launch time


# ---------------------------------------------------------------------------
# 1. Direct @cute.jit calls: RE-TRACES on every call.
#    The implicit cache (keyed on the MLIR hash) skips recompilation, not
#    tracing -- so the meta-stage print fires on every call, and `result`
#    prints as `?` because `b` is a symbolic value during tracing.
# ---------------------------------------------------------------------------
print("=== Direct @cute.jit calls (traces every call) ===\n")
add_dynamicexpr(5.0)
add_dynamicexpr(5.0)

# ---------------------------------------------------------------------------
# 2. Explicit cute.compile (Zero Compile): TRACES ONCE, returns a JIT Executor.
#    The meta-stage print fires exactly once, here. Subsequent executor calls
#    skip MLIR generation and Python tracing entirely.
# ---------------------------------------------------------------------------
print("\n=== cute.compile route (traces once) ===\n")
compiled_add = cute.compile(add_dynamicexpr, cutlass.Float32(0.0))

print("\n--- Compilation done. Running on GPU... ---\n")

compiled_add(5.0)
compiled_add(10.0)
