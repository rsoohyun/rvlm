#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
            

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0), 
            self.lora_B.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True        

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result

class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

# Can Extend to other ones like this

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)
        
        
# soohyun
class LoRAInjectedLinear(nn.Module):
    def __init__(
        self, 
        original_module: nn.Linear, 
        num_lora: int=1,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        **kwargs
    ):
        super().__init__()
        
        self.in_features = original_module.in_features
        self.out_features = original_module.out_features
        use_bias = original_module.bias is not None
        self.org_linear = nn.Linear(self.in_features, self.out_features, bias=use_bias)
        with torch.no_grad():
            self.org_linear.weight.data.copy_(original_module.weight.data)
            if use_bias: self.org_linear.bias.data.copy_(original_module.bias.data)
        
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
            
        self.num_lora = num_lora
        for i in range(self.num_lora):
            self.add_module(f'lora{i}_A', nn.Linear(self.in_features, self.r, bias=False))
            self.add_module(f'lora{i}_B', nn.Linear(self.r, self.out_features, bias=False))
        self.init_lora_params()
        self.used_lora = list(range(self.num_lora))
        self.lora_only = False
        self.scaling = self.lora_alpha / self.r
        
    def init_lora_params(self):
        # initialize B the same way as the default for nn.Linear and A to zero
        # this is different than what is described in the paper but should not affect performance
        for i in range(self.num_lora):
            nn.init.kaiming_uniform_(eval(f'self.lora{i}_A.weight'), a=math.sqrt(5))
            nn.init.zeros_(eval(f'self.lora{i}_B.weight'))

    def forward(self, x: torch.Tensor):
        output = {}
        result = self.org_linear(x) if not self.lora_only else 0
        for i in range(self.num_lora):
            tmp = eval(f'self.lora{i}_A')(x)
            tmp = eval(f'self.lora{i}_B')(tmp)
            output[i] = self.lora_dropout(tmp) * self.scaling
            if i in self.used_lora:
                result += output[i]
        output['org'] = result
        return output


class LoRAInjectedMultiheadAttention(nn.Module):
    def __init__(
        self, 
        original_module: nn.MultiheadAttention, 
        lora_modules: list = ["q","v"],
        num_lora: int=1,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        **kwargs
    ):
        super().__init__()
        
        self.embed_dim = original_module.embed_dim
        self.kdim = original_module.kdim
        self.vdim = original_module.vdim
        self._qkv_same_embed_dim = original_module._qkv_same_embed_dim
        
        self.num_heads = original_module.num_heads
        self.dropout = original_module.dropout
        self.batch_first = original_module.batch_first
        self.head_dim = original_module.head_dim
        
        self.use_bias = original_module.in_proj_bias is not None
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.use_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.kdim, bias=self.use_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.vdim, bias=self.use_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.use_bias)
        
        self.bias_k = original_module.bias_k
        self.bias_v = original_module.bias_v
        self.add_zero_attn = original_module.add_zero_attn
        
        # load existing weights
        with torch.no_grad():
            if original_module.in_proj_weight is not None:
                org_weight = original_module.in_proj_weight.data
                self.q_proj.weight.data.copy_(org_weight[:self.embed_dim,:])
                self.k_proj.weight.data.copy_(org_weight[self.embed_dim:self.embed_dim*2,:])
                self.v_proj.weight.data.copy_(org_weight[self.embed_dim*2:,:])
            else:
                self.q_proj.weight.data.copy_(original_module.q_proj_weight.data)
                self.k_proj.weight.data.copy_(original_module.k_proj_weight.data)
                self.v_proj.weight.data.copy_(original_module.v_proj_weight.data)
            self.out_proj.weight.data.copy_(original_module.out_proj.weight.data)
                
            if self.use_bias:
                org_bias = original_module.in_proj_bias.data
                self.q_proj.bias.data.copy_(org_bias[:self.embed_dim])
                self.k_proj.bias.data.copy_(org_bias[self.embed_dim:self.embed_dim*2])
                self.v_proj.bias.data.copy_(org_bias[self.embed_dim*2:])
                self.out_proj.bias.data.copy_(original_module.out_proj.bias.data)
            
        # apply lora
        self.num_lora = num_lora
        if "q" in lora_modules: self.q_proj = LoRAInjectedLinear(self.q_proj, num_lora, r, lora_alpha, lora_dropout)
        if "k" in lora_modules: self.k_proj = LoRAInjectedLinear(self.k_proj, num_lora, r, lora_alpha, lora_dropout)
        if "v" in lora_modules: self.v_proj = LoRAInjectedLinear(self.v_proj, num_lora, r, lora_alpha, lora_dropout)
        if "out" in lora_modules: self.out_proj = LoRAInjectedLinear(self.out_proj, num_lora, r, lora_alpha, lora_dropout)

    def forward(self, query, key, value, key_padding_mask= None, need_weights= True, attn_mask= None, average_attn_weights=True, is_causal=False):
        why_not_fast_path = ''
        if ((attn_mask is not None and torch.is_floating_point(attn_mask))
           or (key_padding_mask is not None) and torch.is_floating_point(key_padding_mask)):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        
        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))
                
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(self.num_heads, rounding_mode="trunc")
        else:
            head_dim = embed_dim // self.num_heads
            
        q_all = self.q_proj(query)
        k_all = self.k_proj(key)
        v_all = self.v_proj(value)
        
        if not isinstance(q_all, dict): q_all = {i:q_all for i in ['org'] + list(range(self.num_lora))}
        if not isinstance(k_all, dict): k_all = {i:k_all for i in ['org'] + list(range(self.num_lora))}
        if not isinstance(v_all, dict): v_all = {i:v_all for i in ['org'] + list(range(self.num_lora))}
        
        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )
        
        attn_output_all, attn_output_weights_all = {}, {}
        for key in q_all:
            q, k, v = q_all[key], k_all[key], v_all[key]
            
            if self.bias_k is not None and self.bias_v is not None:
                k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
                v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
                if attn_mask is not None:
                    attn_mask = F.pad(attn_mask, (0, 1))
                if key_padding_mask is not None:
                    key_padding_mask = F.pad(key_padding_mask, (0, 1))
            else:
                assert self.bias_k is None
                assert self.bias_v is None
                
            q = q.view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
            k = k.view(k.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)
            v = v.view(v.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)
            if self.add_zero_attn:
                zero_attn_shape = (bsz * self.num_heads, 1, head_dim)
                k = torch.cat(
                    [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1
                )
                v = torch.cat(
                    [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1
                )
                if attn_mask is not None:
                    attn_mask = F.pad(attn_mask, (0, 1))
                if key_padding_mask is not None:
                    key_padding_mask = F.pad(key_padding_mask, (0, 1))
                    
            src_len = k.size(1)
            if key_padding_mask is not None:
                assert key_padding_mask.shape == (
                    bsz,
                    src_len,
                ), f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
                key_padding_mask = (
                    key_padding_mask.view(bsz, 1, 1, src_len)
                    .expand(-1, self.num_heads, -1, -1)
                    .reshape(bsz * self.num_heads, 1, src_len)
                )
                if attn_mask is None:
                    attn_mask = key_padding_mask
                else:
                    attn_mask = attn_mask + key_padding_mask
                    
            if need_weights:
                _B, _Nt, E = q.shape
                q_scaled = q * math.sqrt(1.0 / float(E))
                
                if attn_mask is not None:
                    attn_output_weights = torch.baddbmm(
                        attn_mask, q_scaled, k.transpose(-2, -1)
                    )
                else:
                    attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
                attn_output_weights = F.softmax(attn_output_weights, dim=-1)

                attn_output = torch.bmm(attn_output_weights, v)

                attn_output = (
                    attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
                )
                attn_output = self.out_proj(attn_output)
                if isinstance(attn_output, dict): attn_output = attn_output[key]
                attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

                attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
                if average_attn_weights:
                    attn_output_weights = attn_output_weights.mean(dim=1)

                if not is_batched:
                    attn_output = attn_output.squeeze(1)
                    attn_output_weights = attn_output_weights.squeeze(0)
            else:
                if attn_mask is not None:
                    if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                        attn_mask = attn_mask.unsqueeze(0)
                    else:
                        attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

                q = q.view(bsz, self.num_heads, tgt_len, head_dim)
                k = k.view(bsz, self.num_heads, src_len, head_dim)
                v = v.view(bsz, self.num_heads, src_len, head_dim)

                attn_output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask, 0, is_causal
                )
                attn_output = (
                    attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
                )

                attn_output = self.out_proj(attn_output)
                if isinstance(attn_output, dict): attn_output = attn_output[key]
                attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
                if not is_batched:
                    # squeeze the output if input was unbatched
                    attn_output = attn_output.squeeze(1)
                attn_output_weights = None
            
            if self.batch_first and is_batched:
                attn_output_all[key] = attn_output.transpose(1, 0)
                attn_output_weights_all[key] = attn_output_weights
            else:
                attn_output_all[key] = attn_output
                attn_output_weights_all[key] = attn_output_weights
        
        return attn_output_all, attn_output_weights_all