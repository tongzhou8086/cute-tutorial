import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor):
    print(tensor.layout)  # Prints (?,?):(?,1) for dynamic layout
    cute.printf("tensor: {}", tensor)

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint16)
compiled_func = cute.compile(foo, a)
compiled_func(a)

b = torch.tensor([[11, 12], [13, 14], [15, 16]], dtype=torch.uint16)
compiled_func(b)  # Reuse the same compiled function for different shape


## Static Layout

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
a_pack = from_dlpack(a)
compiled_func = cute.compile(foo, a_pack)
compiled_func(a_pack)