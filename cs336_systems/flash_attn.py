import torch
from torch import Tensor
from jaxtyping import Float, Bool, Int
import triton.language as tl
from einops import rearrange, einsum
import pytest

def cdiv(x: int, y: int):
    return (x + y - 1) // y

def flash_attn_forward(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    causal: bool = False
) -> tuple[Float[Tensor, " ... queries d_v"], Float[Tensor, "... queries"]]:
        assert len(Q.shape) == 4
        assert len(K.shape) == 4
        assert len(V.shape) == 4

        batch_size = Q.shape[0]
        num_heads = Q.shape[1]
        seq_len = Q.shape[2]
        d_k = Q.shape[3]
        d_v = V.shape[3]

        softmax_scale = 1. / (d_k**0.5)
        O = torch.zeros((batch_size, num_heads, seq_len, d_v), dtype=torch.float32)
        L = torch.zeros((batch_size, num_heads, seq_len, 1), dtype=torch.float32)
        # tileQ = 16
        tileQ = seq_len
        tileK = seq_len

        for b in range(batch_size):
            for h in range(num_heads):
                for q_chunk in range(cdiv(seq_len, tileQ)):
                    startI = q_chunk * tileQ
                    endI = min((q_chunk+1)*tileQ, seq_len)
                    sliceI = (b, h, slice(startI, endI), slice(None))
                    m = torch.full((tileQ, 1), fill_value=-torch.inf, dtype=torch.float32)
                    l = torch.zeros((tileQ, 1), dtype=torch.float32)
                    Qi = Q[sliceI]
                    for kv_chunk in range(cdiv(seq_len, tileK)):
                        # Load K,V
                        startJ = kv_chunk * tileK
                        endJ = min((kv_chunk+1)*tileK, seq_len)
                        sliceJ = (b, h, slice(startJ, endJ), slice(None))
                        Kj = K[sliceJ]
                        Vj = V[sliceJ]
                        # Compute QK
                        Sij = einsum(Qi, Kj, "query d_k, key d_k -> query key") * softmax_scale
                        # Compute new row-max
                        m_cur = torch.max(Sij, dim=-1)[0].reshape(tileQ, 1)
                        m_new = torch.max(torch.stack( (m_cur, m) , dim=-1), dim=-1, keepdim=False)[0]
                        Pij = torch.exp(Sij - m_new)
                        # Compute row-sum
                        l_cur = torch.sum(Pij, dim=-1).reshape(tileQ, 1)
                        l_new = torch.exp(m - m_new)*l + l_cur
                        PV = einsum(Pij, Vj, "query key, key d_v -> query d_v")
                        # Update O
                        O[sliceI] = torch.exp(m - m_new) * O[sliceI] + PV

                        m = m_new
                        l = l_new

                    O[sliceI] /= l
                    L[b, h, startI:endI] = m + torch.log(l)

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
