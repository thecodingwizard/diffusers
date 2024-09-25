import torch
from torch import nn
import triton
import triton.language as tl
from triton.language.extra.libdevice import rsqrt

from typing import Optional, Union, Tuple

import flashattn_hopper_cuda

torch.manual_seed(0)

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    cos, sin = freqs_cis  # [S, D]
    cos, sin = cos.to(x.device), sin.to(x.device)

    # Used for flux, cogvideox, hunyuan-dit
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)

    out = (x.float() * cos[None,:,None,:] + x_rotated.float() * sin[None,:,None,:]).to(x.dtype)

    return out

class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, int):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.randn(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states

norm_q = RMSNorm(128, eps=1e-5).to("cuda").to(torch.bfloat16)
norm_k = RMSNorm(128, eps=1e-5).to("cuda").to(torch.bfloat16)
to_qkv = torch.nn.Linear(3072, 3072*3, device="cuda", dtype=torch.bfloat16)

torch.library.define(
    "mylib::flash_attn_with_out",
    "(Tensor q, Tensor k, Tensor v, Tensor out) -> Tensor",
)

@torch.library.impl("mylib::flash_attn_with_out", "cuda")
def custom_func(q, k, v, out):
    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert v.stride(-1) == 1
    assert out.stride(-1) == 1 # not sure if this is necessary
    softmax_scale = q.shape[-1] ** (-0.5)
    causal = False
    window_size = (-1, -1)
    out, q, k, v, out_padded, softmax_lse, S_dmask = flashattn_hopper_cuda.fwd(
        q,
        k,
        v,
        out,
        softmax_scale,
        None, # descale q/k/v
        None,
        None,
        causal,
        window_size[0],
        window_size[1],
    )
    return out

@torch.library.register_fake("mylib::flash_attn_with_out")
def custom_func_abstract(q, k, v, out):
    return out


# @torch.compile
def reference_attention(
    hidden_states: torch.FloatTensor,
    image_rotary_emb: Tuple[torch.Tensor],
    out: torch.Tensor,
    to_qkv,
    norm_q,
    norm_k
) -> torch.FloatTensor:
    batch_size, _, _ = hidden_states.shape

    # `sample` projections.
    qkv = to_qkv(hidden_states)
    split_size = qkv.shape[-1] // 3
    query, key, value = torch.split(qkv, split_size, dim=-1)

    inner_dim = key.shape[-1]
    # n_heads = 24
    head_dim = inner_dim // 24

    # [B, S, H, D]
    query = query.view(batch_size, -1, 24, head_dim)
    key = key.view(batch_size, -1, 24, head_dim)
    value = value.view(batch_size, -1, 24, head_dim)

    query = norm_q(query)
    key = norm_k(key)

    query = apply_rotary_emb(query, image_rotary_emb)
    key = apply_rotary_emb(key, image_rotary_emb)

    hidden_states = torch.nn.functional.scaled_dot_product_attention(query.transpose(1,2), key.transpose(1,2), value.transpose(1,2), dropout_p=0.0, is_causal=False)
    hidden_states = hidden_states.transpose(1, 2)
    hidden_states = hidden_states.reshape(batch_size, -1, 3072)

    out.copy_(hidden_states)

    # hidden_states = torch.ops.mylib.custom_func(query, key, value, out)
    # hidden_states = hidden_states.reshape(batch_size, -1, 24 * head_dim)

    return out


@triton.autotune(
  configs=[
    triton.Config(kwargs={}, num_warps=1),
    triton.Config(kwargs={}, num_warps=2),
    triton.Config(kwargs={}, num_warps=4),
    triton.Config(kwargs={}, num_warps=8),
  ],
  key=[]
)
@triton.jit
def _new_fused_qkv_rmsnorm_embed(
    X_ptr,
    X_stride_1,
    X_stride_2,
    W_q_ptr,
    W_k_ptr,
    eps,

    freq_cos_ptr,
    freq_sin_ptr, # assumed to be contiguous. [4224, 128]
):
    """
    y_i = (x_i / (RMS)) * (offset + wi), RMS = sqrt(sum(x_i^2) / N)

    Reference:
    1. https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    2. https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/rms_layernorm.py#L22
    3. https://arxiv.org/pdf/1910.07467
    """

    n_heads: tl.constexpr = 24 * 2 # query, then key
    head_dim: tl.constexpr = 128

    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, head_dim)
    mask = col_offsets < head_dim

    X_ptr += row_idx * X_stride_1
    W_q_row = tl.load(W_q_ptr + col_offsets, mask=mask, other=0)
    W_q_row = W_q_row.to(tl.float32)
    W_k_row = tl.load(W_k_ptr + col_offsets, mask=mask, other=0)
    W_k_row = W_k_row.to(tl.float32)

    cos_first_half = tl.load(freq_cos_ptr + row_idx * 128 + tl.arange(0, head_dim // 2) * 2).to(tl.float32)
    cos_second_half = tl.load(freq_cos_ptr + row_idx * 128 + tl.arange(0, head_dim // 2) * 2 + 1).to(tl.float32)

    sin_first_half = tl.load(freq_sin_ptr + row_idx * 128 + tl.arange(0, head_dim // 2) * 2).to(tl.float32) * -1
    sin_second_half = tl.load(freq_sin_ptr + row_idx * 128 + tl.arange(0, head_dim // 2) * 2 + 1).to(tl.float32)

    for i in range(n_heads):
        X_row = tl.load(X_ptr + i * X_stride_2 + col_offsets, mask=mask, other=0)
        X_row = X_row.to(tl.float32)

        mean_square = tl.sum(X_row * X_row, axis=0) / head_dim
        rstd = rsqrt(mean_square + eps)

        X_row = X_row * rstd

        # lol ugh apparently llama does this calculation in bf16, but maybe f32 is OK?
        # X_row = X_row.to(tl.bfloat16)
        Y_row = X_row * (W_q_row if i < 24 else W_k_row)

        # tl.store(X_ptr + i * X_stride_2 + tl.arange(0, 128), Y_row.to(tl.float16))
        # --- ^ end RMS norm
        # --- v begin rope
        # Y_row is [1, 128] right now
        # X is [(B S) H D]. one program ID corresponds to one of (B S)
        # cos, sin = freqs_cis  # [S, D]
        # x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        # x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        # out = (x.float() * cos[None,:,None,:] + x_rotated.float() * sin[None,:,None,:]).to(x.dtype)

        Y_real, Y_imag = Y_row.reshape(64, 2).split()

        Y_first_half = Y_real * cos_first_half + Y_imag * sin_first_half
        Y_second_half = Y_imag * cos_second_half + Y_real * sin_second_half
        tl.store(X_ptr + i * X_stride_2 + tl.arange(0, head_dim//2) * 2, Y_first_half.to(tl.bfloat16))
        tl.store(X_ptr + i * X_stride_2 + tl.arange(0, head_dim//2) * 2 + 1, Y_second_half.to(tl.bfloat16))


def my_attention(
    hidden_states: torch.FloatTensor,
    image_rotary_emb: Tuple[torch.Tensor],
    out: torch.Tensor,
    qkv=None,
    norm_q=norm_q,
    norm_k=norm_k,
) -> torch.FloatTensor:
    batch_size, _, _ = hidden_states.shape

    # `sample` projections.
    if qkv is None:
        qkv = to_qkv(hidden_states)
    split_size = qkv.shape[-1] // 3
    query, key, value = torch.split(qkv, split_size, dim=-1)

    inner_dim = key.shape[-1]
    # n_heads = 24
    head_dim = inner_dim // 24

    query = query.view(-1, 24, head_dim)
    key = key.view(-1, 24, head_dim)

    _new_fused_qkv_rmsnorm_embed[(query.shape[0],)](
        query,
        query.stride(0),
        query.stride(1),
        norm_q.weight,
        norm_k.weight,
        1e-6,
        image_rotary_emb[0],
        image_rotary_emb[1],
    )

    # [B, S, H, D]
    query = query.view(batch_size, -1, 24, head_dim)
    key = key.view(batch_size, -1, 24, head_dim)
    value = value.view(batch_size, -1, 24, head_dim)
    out = out.view(batch_size, -1, 24, head_dim)

    hidden_states = torch.ops.mylib.flash_attn_with_out(query, key, value, out)
    hidden_states = hidden_states.reshape(batch_size, -1, 24 * head_dim)

    return hidden_states


def initialize():
    # my_attention modifies in place, and the first time autotune is run it will run many times
    hidden_states = torch.randn((1, 4224, 3072), dtype=torch.bfloat16, device="cuda")
    image_rotary_emb = (torch.randn((4224, 128), dtype=torch.bfloat16, device="cuda"), torch.randn((4224, 128), dtype=torch.bfloat16, device="cuda"))

    out = torch.empty((1, 4224, 3072 + 3072 * 4), dtype=torch.bfloat16, device="cuda")
    attn_out = out[:,:,:3072]
    mlp_out = out[:,:,3072:]

    # reference_attention(hidden_states, image_rotary_emb)
    my_attention(hidden_states, image_rotary_emb, attn_out)


def verify(ref, my):
    hidden_states = torch.randn((1, 4224, 3072), dtype=torch.bfloat16, device="cuda")
    image_rotary_emb = (torch.randn((4224, 128), dtype=torch.bfloat16, device="cuda"), torch.randn((4224, 128), dtype=torch.bfloat16, device="cuda"))

    out = torch.empty((1, 4224, 3072 + 3072 * 4), dtype=torch.bfloat16, device="cuda")
    attn_out = out[:,:,:3072]
    mlp_out = out[:,:,3072:]

    ref_out = ref(hidden_states, image_rotary_emb, attn_out)
    my_out = my(hidden_states, image_rotary_emb, attn_out)
    assert my_out.allclose(attn_out)
    assert my_out.data_ptr() == attn_out.data_ptr()

    print((ref_out - my_out).abs().max())
    print((ref_out - my_out).abs().mean())


def bench(fn):
    warmup_iters, bench_iters = 100, 1000
    hidden_states = torch.randn((1, 4224, 3072), dtype=torch.bfloat16, device="cuda")
    image_rotary_emb = (torch.randn((4224, 128), dtype=torch.bfloat16, device="cuda"), torch.randn((4224, 128), dtype=torch.bfloat16, device="cuda"))

    out = torch.empty((1, 4224, 3072 + 3072 * 4), dtype=torch.bfloat16, device="cuda")
    attn_out = out[:,:,:3072]
    mlp_out = out[:,:,3072:]

    for i in range(warmup_iters):
        fn(hidden_states, image_rotary_emb, attn_out)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(bench_iters):
        fn(hidden_states, image_rotary_emb, attn_out)
    end.record()
    torch.cuda.synchronize()

    print(f"Elapsed time: {start.elapsed_time(end) / bench_iters * 1000}us")

if __name__ == "__main__":
    """
    1914.39111328125us uncompiled
    838.5963745117188us compiled
    753.7843322753906us compiled if you decrease number of iterations??

    Name	Duration	GPU	Context
    linear  340.188 μs	GPU 0	Stream 7
    triton_	106.207 μs	GPU 0	Stream 7
    attn	363.452 μs	GPU 0	Stream 7

    transposing makes no difference

    Name	Duration	GPU	Context
    linear	366.204 μs	GPU 0	Stream 7
    mine	49.311 μs	GPU 0	Stream 7
    attn	391.421 μs	GPU 0	Stream 7

    lol these numbers are highly sus. I think the GPU overheats or something and slows down if you do more iterations??? idk
    769.697509765625us custom
    """
    initialize()
    verify(reference_attention, my_attention)
    # bench(reference_attention)
    bench(my_attention)
