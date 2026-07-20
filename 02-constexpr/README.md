# Static argument vs. Dynamic argument

CuTe DSL supports both static and dynamic arguments for JIT functions.

- Static arguments hold values that are known at compile time. It is not included in the generated JIT function signature.
- Dynamic arguments hold values that are only known at runtime.

# Inspect generated MLIR

Run `CUTE_DSL_KEEP=ir python 02-constexpr/main.py` and we will get a file called `cute_dsl_cutlass_foo_2_2_clean.mlir` with
the following content:

```
module attributes {gpu.container_module} {
  func.func @cutlass_foo_2_2(%arg0: i32) -> i32 attributes {llvm.emit_c_interface} {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    cute.print("x: {}\0A", %arg0) : i32
    cute.print("y: {}\0A", %c2_i32) : i32
    return %c0_i32 : i32
  }
}
```

We can see that `y` - the constexpr argument becomes a hard-coded constant `2` in the generated IR while `x` - the dynamic
argument stays as a regular function argument (`%arg0`).

# Trace-time branch selection

`trace_constexpr_dynamicexpr_ir.py` instruments the same add example as the
JIT chapter, but with the parameter marked as `cutlass.Constexpr` and the local
typed literal marked with `cutlass.const_expr(cutlass.Float32(2.0))`. The
generated IR bakes in the value instead of passing it as a runtime `%arg0`.

`trace_if_constexpr_ir.py` instruments a branch guarded by `cutlass.const_expr`.
Only the selected branch is traced and emitted; the other branch does not appear
in the generated MLIR.

Run:

```
python 02-constexpr/trace_constexpr_dynamicexpr_ir.py
python 02-constexpr/trace_if_constexpr_ir.py
```
