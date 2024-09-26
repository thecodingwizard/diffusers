# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...models.attention import FeedForward
from ...models.attention_processor import (
    Attention,
    AttentionProcessor,
    FluxAttnProcessor2_0,
    FusedFluxAttnProcessor2_0,
)
from ...models.modeling_utils import ModelMixin
from ...models.normalization import AdaLayerNormContinuous, AdaLayerNormZero
from ...utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from ...utils.torch_utils import maybe_allow_in_graph
from ..embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from ..modeling_outputs import Transformer2DModelOutput

import triton
import triton.language as tl
from triton.language.extra.libdevice import rsqrt
from . import triton_attn
from . import triton_gelu_copy


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



@triton.jit
def _layer_norm_forward_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    shift_msa,
    scale_msa,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    X_ptr += row_idx * X_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    mean = tl.sum(X_row, axis=0) / n_cols
    var = tl.sum((X_row - mean) * (X_row - mean), axis=0) / n_cols
    rstd = rsqrt(var + eps)

    shift = tl.load(shift_msa + col_offsets, mask=mask, other=0).to(tl.float32)
    scale = tl.load(scale_msa + col_offsets, mask=mask, other=0).to(tl.float32)

    Y_row = (X_row - mean) * rstd
    Y_row = Y_row * (1.0 + scale) + shift
    tl.store(Y_ptr + col_offsets + row_idx * Y_stride, Y_row.to(tl.bfloat16), mask=mask)

@triton.jit
def _modulation_kernel(
    embed_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    col_idx = tl.program_id(0)

    weight_ptr += col_idx * 3072
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < 3072

    embed = tl.load(embed_ptr + offsets, mask=mask).to(tl.float32)

    sigmoid_x = 1 / (1 + tl.exp(-embed))
    embed_after_silu = embed * sigmoid_x

    weight = tl.load(weight_ptr + offsets, mask=mask).to(tl.float32)
    bias = tl.load(bias_ptr + col_idx).to(tl.float32)

    out = tl.sum(embed_after_silu * weight) + bias
    tl.store(out_ptr + col_idx, out.to(tl.bfloat16))


@torch.library.custom_op("mylib::modulation_kernel", mutates_args={"out"})
def modulation_kernel(embed: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, out: torch.Tensor) -> None:
    _modulation_kernel[(3072*3,)](
        embed,
        weight,
        bias,
        out,
        BLOCK_SIZE=4096,
        num_warps=8,
    )

@torch.library.custom_op("mylib::ln_forward", mutates_args={"actual_out"})
def ln_forward_op(x: torch.Tensor, actual_out: torch.Tensor, shift_msa: torch.Tensor, scale_msa: torch.Tensor) -> None:
    n_rows = x.shape[1]
    dim = x.shape[-1]
    _layer_norm_forward_kernel[(n_rows,)](
        x,
        x.stride(1), # since first dimension is empty
        actual_out,
        actual_out.stride(1), # since first dimension is empty
        shift_msa,
        scale_msa,
        dim,
        eps=1e-6,
        BLOCK_SIZE=4096, # https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/utils.py
        num_warps=8,
    )


class AdaLayerNormZeroSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.

    
    optimized: 2160.682373046875us, 2167.112548828125us
    unoptimized: 2191.817138671875us, 2200.03515625us
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

        self.emb_out = torch.empty((1, 3072*3), dtype=torch.bfloat16, device="cuda")
        self.actual_out = torch.empty((1, 4224, 3072), dtype=torch.bfloat16, device="cuda")

    # @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.ops.mylib.modulation_kernel(emb, self.linear.weight, self.linear.bias, self.emb_out)
        # _modulation_kernel[(3072*3,)](
        #     emb,
        #     self.linear.weight,
        #     self.linear.bias,
        #     self.emb_out,
        #     BLOCK_SIZE=4096,
        #     num_warps=8,
        # )

        shift_msa, scale_msa, gate_msa = self.emb_out.chunk(3, dim=1)

        # n_rows = x.shape[1]
        # dim = x.shape[-1]
        # _layer_norm_forward_kernel[(n_rows,)](
        #     x,
        #     x.stride(1), # since first dimension is empty
        #     self.actual_out,
        #     self.actual_out.stride(1), # since first dimension is empty
        #     shift_msa,
        #     scale_msa,
        #     dim,
        #     eps=1e-6,
        #     BLOCK_SIZE=4096, # https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/utils.py
        #     num_warps=8,
        # )
        torch.ops.mylib.ln_forward(x, self.actual_out, shift_msa, scale_msa)
        return self.actual_out, gate_msa


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        processor = FluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        attn_and_mlp_out = torch.empty((1, 4224, 3072 + 3072 * 4), dtype=torch.bfloat16, device="cuda")
        attn_out = attn_and_mlp_out[:,:,:3072]
        mlp_out = attn_and_mlp_out[:,:,3072:]

        # gemm_mod_bf16.run_aten(norm_hidden_states.squeeze(0), self.proj_mlp.weight.T, mlp_out.squeeze(0), self.proj_mlp.bias)

        # gemm_mod_bf16.run(norm_hidden_states, self.proj_mlp.weight, mlp_out)
        # mlp_out.add_(self.proj_mlp.bias)
        # mlp_out_tmp = torch.nn.functional.gelu(mlp_out, approximate="tanh") # warning: not in place
        # mlp_out.copy_(mlp_out_tmp) # temporary
        qkv_mlp = self.qkv_mlp_linear(norm_hidden_states)
        qkv, mlp_hidden_states = torch.split(qkv_mlp, [3 * 3072, self.mlp_hidden_dim], dim=-1)

        # triton_gelu_copy.gelu_and_copy(mlp_hidden_states, mlp_out)

        torch.ops.mylib.gelu_and_copy(mlp_hidden_states, mlp_out)

        triton_attn.my_attention(norm_hidden_states, image_rotary_emb, out=attn_out, qkv=qkv, norm_q = self.attn.norm_q, norm_k = self.attn.norm_k)
        # attn_out_2 = self.attn(
        #     hidden_states=norm_hidden_states,
        #     image_rotary_emb=image_rotary_emb
        # )
        # attn_out.copy_(attn_out_2)

        gate = gate.unsqueeze(1)
        hidden_states = self.fuse_residual(self.proj_out(attn_and_mlp_out), gate, residual)
        # hidden_states = gate * self.proj_out(attn_and_mlp_out)
        # hidden_states = residual + hidden_states

        return hidden_states


    @torch.compile
    def fuse_residual(self, proj_out, gate, hidden_states):
        return gate * proj_out + hidden_states

    def fuse_projections(self):
        self.qkv_mlp_linear = nn.Linear(3072, 3 * 3072 + self.mlp_hidden_dim, device="cuda", dtype=torch.bfloat16)
        concatenated_weights = torch.cat([self.attn.to_qkv.weight, self.proj_mlp.weight])
        concatenated_bias = torch.cat([self.attn.to_qkv.bias, self.proj_mlp.bias])
        self.qkv_mlp_linear.weight.copy_(concatenated_weights)
        self.qkv_mlp_linear.bias.copy_(concatenated_bias)



# @maybe_allow_in_graph
# class FluxSingleTransformerBlock(nn.Module):
#     r"""
#     A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

#     Reference: https://arxiv.org/abs/2403.03206

#     Parameters:
#         dim (`int`): The number of channels in the input and output.
#         num_attention_heads (`int`): The number of heads to use for multi-head attention.
#         attention_head_dim (`int`): The number of channels in each head.
#         context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
#             processing of `context` conditions.
#     """

#     def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
#         super().__init__()
#         self.mlp_hidden_dim = int(dim * mlp_ratio)

#         self.norm = AdaLayerNormZeroSingle(dim)
#         self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
#         self.act_mlp = nn.GELU(approximate="tanh")
#         self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

#         processor = FluxAttnProcessor2_0()
#         self.attn = Attention(
#             query_dim=dim,
#             cross_attention_dim=None,
#             dim_head=attention_head_dim,
#             heads=num_attention_heads,
#             out_dim=dim,
#             bias=True,
#             processor=processor,
#             qk_norm="rms_norm",
#             eps=1e-6,
#             pre_only=True,
#         )

#     def forward(
#         self,
#         hidden_states: torch.FloatTensor,
#         temb: torch.FloatTensor,
#         image_rotary_emb=None,
#     ):
#         residual = hidden_states
#         norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
#         mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

#         attn_output = my_attention(norm_hidden_states, image_rotary_emb, self.attn.to_qkv, self.attn.norm_q.weight, self.attn.norm_k.weight)

#         hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
#         gate = gate.unsqueeze(1)
#         hidden_states = gate * self.proj_out(hidden_states)
#         hidden_states = residual + hidden_states
#         if hidden_states.dtype == torch.float16:
#             hidden_states = hidden_states.clip(-65504, 65504)

#         return hidden_states

@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    @torch.compile
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedFluxAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        with torch.inference_mode():
            for module in self.modules():
                if isinstance(module, FluxSingleTransformerBlock):
                    module.fuse_projections()

        self.set_attn_processor(FusedFluxAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @torch.compile
    def forward_compiled_section(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        profile=False
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        return hidden_states, temb, image_rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        profile=False
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
                Actually, I think this is batch x # patches x # channels?
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        hidden_states, temb, image_rotary_emb = self.forward_compiled_section(hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, txt_ids, guidance, joint_attention_kwargs, controlnet_block_samples, controlnet_single_block_samples, return_dict)

        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        return self.forward_end(hidden_states, temb, return_dict)
        
    
    @torch.compile
    def forward_end(self, hidden_states, temb, return_dict):
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            assert False
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
