# Dynamic and Static Layout

Running the `main.py` produces the following output:

```
(base) ➜  cute-tutorial git:(main) ✗ python 03-torch-tensors/main.py
(?,?):(?{i64},1)
tensor: raw_ptr(0x0000561885b1e4c0: i16, generic, align<2>) o (2,2):(2,1)
tensor: raw_ptr(0x00005618864250c0: i16, generic, align<2>) o (3,2):(2,1)
(3):(1)
tensor: raw_ptr(0x00005618863e1640: i16, generic, align<2>) o (3):(1) = 
  ( 1, 2, 3 )
```

# What Does (2,2):(2,1) Mean?

In CuTe DSL, the colon `:` does not separate dimensions like rows and columns. Instead, it separates the **Shape** from the **Stride**.

Every layout in CuTe is printed using this exact formula:
**`(Shape) : (Stride)`**

Here is exactly what those prints mean, broken down piece by piece.

## 1. The 1D Static Layout: `(3):(1)`

You guessed that this might mean a shape of `(3, 1)`, but it actually represents a 1-dimensional tensor.

* **Shape `(3)`:** The tensor has exactly 3 elements.
* **Stride `(1)`:** To get to the next element in memory, you move forward by 1 position. This means the memory is perfectly contiguous.

When you passed `a_pack = from_dlpack([1, 2, 3])`, CuTe hardcoded both the shape (3) and the stride (1) directly into the compiled kernel.

## 2. The 2D Tensor `b`: `(3,2):(2,1)`

When you passed `b = torch.tensor([[11, 12], [13, 14], [15, 16]])`, CuTe printed `(3,2):(2,1)` at runtime.

* **Shape `(3, 2)`:** This perfectly matches PyTorch. You have 3 rows and 2 columns.
* **Stride `(2, 1)`:** This tells the GPU exactly how to navigate the raw 1D memory array to find the 2D data.
* The `2` means: To move down one **row**, jump forward 2 elements in memory. (Because each row has 2 columns).
* The `1` means: To move right one **column**, jump forward 1 element in memory.



This `(2, 1)` stride is the classic signature of **Row-Major** (C-contiguous) memory, which is PyTorch's default memory format.

## 3. The Meta-Stage Dynamic Print: `(?,?):(?{i64},1)`

When you ran `cute.compile(foo, a)` using the raw PyTorch tensor, it traced the Python code and printed `(?,?):(?{i64},1)`. Here is how the JIT compiler interpreted your PyTorch tensor during tracing:

* **Shape `(?, ?)`:** The compiler sees you passed a 2D tensor, but because you didn't use `from_dlpack`, it replaces the actual dimensions (2x2) with symbolic question marks. This tells the kernel to expect *any* 2D shape.
* **Stride `(?{i64}, 1)`:**
* The row stride is `?` (as a 64-bit integer) because the row stride depends on how many columns there are, which is unknown at compile time.
* The column stride is `1` because CuTe assumes the innermost dimension of a standard PyTorch tensor is contiguous in memory.



Because this layout was compiled using `?` symbols, it successfully processed `a` (shape `2,2`, stride `2,1`) and then immediately reused the exact same binary to process `b` (shape `3,2`, stride `2,1`).