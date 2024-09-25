import torch

import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh


@triton.jit
def gelu_and_copy_kernel(
    a_ptr,
    b_ptr,
    N,
    stride_a,
    stride_b,
):
    """
    b = gelu(a)
    """
    BLOCK_SIZE: tl.constexpr = 1024
    row = tl.program_id(0)

    for i in range(0, tl.cdiv(N, BLOCK_SIZE)):
        offs = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offs < N
        a_row = tl.load(a_ptr + stride_a * row + offs, mask=mask).to(tl.float32)

        b_row = gelu(a_row).to(tl.bfloat16)

        tl.store(b_ptr + stride_b * row + offs, b_row, mask=mask)
    

@triton.jit
def gelu(
    a_row,
):
    # tanh approximation form of GELU is computed with:
    # 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    
    return geglu_a

def gelu_and_copy(a, b):
    gelu_and_copy_kernel[(a.shape[-2],)](
        a, b, a.shape[-1], a.stride(-2), b.stride(-2)
    )
    return b


if __name__ == "__main__":
    # %%
    # Unit Test
    # ---------
    #
    # We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).

    M, N = 4224, 3072 * 4

    torch.manual_seed(0)
    a = torch.randn((M, N), device='cuda', dtype=torch.bfloat16)
    attn_and_mlp_out = torch.empty((1, 4224, 3072 + N), dtype=torch.bfloat16, device="cuda")
    attn_out = attn_and_mlp_out[:,:,:3072]
    b = attn_and_mlp_out[:,:,3072:]
    triton_output = gelu_and_copy(a, b)
    torch_output = torch.nn.functional.gelu(a, approximate="tanh")
    # print(f"triton_output_with_fp16_inputs={triton_output}")
    # print(f"torch_output_with_fp16_inputs={torch_output}")
    print((triton_output - torch_output).abs().max())
    print((triton_output - torch_output).abs().mean())
    rtol = 0
    if torch.allclose(triton_output, torch_output, atol=5e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")



    a = torch.randn((M, N), device='cuda', dtype=torch.bfloat16)

    attn_and_mlp_out = torch.empty((1, 4224, 3072 + 3072 * 4), dtype=torch.bfloat16, device="cuda")
    attn_out = attn_and_mlp_out[:,:,:3072]
    b = attn_and_mlp_out[:,:,3072:]

    quantiles = [0.5, 0.2, 0.8]

    ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.gelu(a, approximate="tanh"), quantiles=quantiles, warmup=100, rep=1000)
    print(f"torch :\t{min_ms*1000}us\t{ms*1000}us\t{max_ms*1000}us")

    ms, min_ms, max_ms = triton.testing.do_bench(lambda: gelu_and_copy(a, b), quantiles=quantiles, warmup=100, rep=1000)
    print(f"triton:\t{min_ms*1000}us\t{ms*1000}us\t{max_ms*1000}us")
