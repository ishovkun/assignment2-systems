import torch
from torch import Tensor
from jaxtyping import Float, Bool, Int
import triton.language as tl
from einops import rearrange, einsum
from math import sqrt
import pytest

def cdiv(x: int, y: int):
    return (x + y - 1) // y

def flash_attn_forward(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    causal: bool = False) -> tuple[Float[Tensor, " ... queries d_v"], Float[Tensor, "... queries"]]:
        batch_size = Q.shape[0]
        num_heads = Q.shape[1]
        seq_len = Q.shape[2]
        d_k = Q.shape[3]
        # d_v = V.shape[-1]

        softmax_scale = 1. / sqrt(d_k)
        O = torch.zeros_like(Q)
        L = torch.zeros((batch_size, num_heads, seq_len, 1), dtype=torch.float32)
        tileQ = 1 << 4
        tileK = 1 << 4

        for batch in range(batch_size):
            for head in range(num_heads):
                for q_chunk in range(cdiv(seq_len, tileQ)):
                    startI = q_chunk * tileQ
                    endI = min((q_chunk+1)*tileQ, seq_len)
                    Qi = Q[batch, head, startI:endI, :]
                    m = torch.full((tileQ, 1), fill_value=-torch.inf, dtype=torch.float32)
                    l = torch.zeros((tileQ, 1), dtype=torch.float32)
                    for kv_chunk in range(cdiv(seq_len, tileK)):
                        startJ = kv_chunk * tileK
                        endJ = min((kv_chunk+1)*tileK, seq_len)
                        Kj = K[batch, head, startJ:endJ, :]
                        Vj = V[batch, head, startJ:endJ, :]

                        Sij = einsum(Qi, Kj, "query d_k, key d_k -> query key")
                        Sij *= softmax_scale
                        m_cur = torch.max(Sij, dim=-1)[0].reshape(tileQ, 1)

                        Pij = torch.exp(Sij - m_cur)
                        l_cur = torch.sum(Sij, dim=-1).reshape(tileQ, 1)

                        m_new = torch.max(torch.stack( (m_cur, m) , dim=-1), dim=-1, keepdim=False)[0]
                        l_new = l_cur * torch.exp(m - m_new) + l * torch.exp(m_cur - m_new)

                        PV = Pij @ Vj

                        O[batch, head, startI:endI,:] = \
                                ((l * torch.exp(m - m_new) * O[batch, head, startI:endI,:]) +
                                        (torch.exp(m_cur - m_new) * PV)) / l_new

                        m = m_new
                        l = l_new
                        nan_count = torch.sum(torch.isnan(O))
                        if (nan_count > 0):
                            print()
                            print(f"{nan_count} nans found")
                            print(q_chunk)
                            pytest.exit(0)
                    L[batch, head, startI:endI] = l

            return O, L

class FlashAttentionTorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys    d_k"],
        V: Float[Tensor, " ... keys    d_v"],
        causal: bool = False
    ) -> Float[Tensor, " ... queries d_v"]:
        rearrange_back = False
        if len(Q.shape) == 3:
            Q = rearrange(Q, "heads seq d -> 1 heads seq d")
            K = rearrange(K, "heads seq d -> 1 heads seq d")
            V = rearrange(V, "heads seq d -> 1 heads seq d")
            rearrange_back = True
        O, L = flash_attn_forward(Q, K, V, causal)
        if rearrange_back:
            L = rearrange(L, "1 heads seq 1 -> heads seq")
            O = rearrange(O, "1 heads seq dim -> heads seq dim")

        ctx.save_for_backward(O, L)
        return O


    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("Backward pass not implemented")
