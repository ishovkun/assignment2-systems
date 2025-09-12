import torch
from torch import Tensor
from jaxtyping import Float, Bool, Int
import triton
import triton.language as tl
from einops import rearrange, einsum
import pytest

def cdiv(x: int, y: int):
    return (x + y - 1) // y

def flash_attn_forward(
    Q: Float[Tensor, "batch queries d_k"],
    K: Float[Tensor, "batch keys    d_k"],
    V: Float[Tensor, "batch keys    d_v"],
    causal: bool = False
) -> tuple[Float[Tensor, " ... queries d_v"], Float[Tensor, "... queries"]]:
    assert len(Q.shape) == 3
    assert len(K.shape) == 3
    assert len(V.shape) == 3

    batch_dim = Q.shape[0]
    seq_len = Q.shape[1]
    d_k = Q.shape[2]
    d_v = V.shape[2]

    softmax_scale = 1. / (d_k**0.5)
    O = torch.zeros((batch_dim, seq_len, d_v), dtype=torch.float32)
    L = torch.zeros((batch_dim, seq_len, 1), dtype=torch.float32)
    tileQ = seq_len
    tileK = seq_len

    for b in range(batch_dim):
        for q_chunk in range(cdiv(seq_len, tileQ)):
            startI = q_chunk * tileQ
            endI = min((q_chunk+1)*tileQ, seq_len)
            sliceI = (b, slice(startI, endI), slice(None))
            m = torch.full((tileQ, 1), fill_value=-torch.inf, dtype=torch.float32)
            l = torch.zeros((tileQ, 1), dtype=torch.float32)
            Qi = Q[sliceI]
            for kv_chunk in range(cdiv(seq_len, tileK)):
                # Load K,V
                startJ = kv_chunk * tileK
                endJ = min((kv_chunk+1)*tileK, seq_len)
                sliceJ = (b, slice(startJ, endJ), slice(None))
                Kj = K[sliceJ]
                Vj = V[sliceJ]
                # Compute QK
                Sij = einsum(Qi, Kj, "query d_k, key d_k -> query key") * softmax_scale
                # Compute new row-max
                m_cur = torch.max(Sij, dim=-1)[0].reshape(tileQ, 1)
                m_new = torch.max(torch.stack( (m_cur, m) , dim=-1), dim=-1, keepdim=False)[0]
                # m_new = torch.maximum(m_cur, m , dim=-1)
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
            L[b, startI:endI] = m + torch.log(l)

    return O, L

@triton.jit
def flash_attn_forward_triton(
    q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
    batch_dim, seq_len,
    qk_stride_batch, qk_stride_seq, qk_stride_d,
    v_stride_batch, v_stride_seq, v_stride_d,
    o_stride_batch, o_stride_seq, o_stride_d,
    l_stride_batch, l_stride_seq,
    tile_size_q: tl.constexpr, tile_size_kv: tl.constexpr,
    max_head_dim: tl.constexpr
):
    batch = tl.program_id(1) # blockIdx.y
    query_tile_idx = tl.program_id(0)
    scale = 1. / max_head_dim**0.5

    q_block_ptr = tl.make_block_ptr(
        q_ptr + batch * qk_stride_batch,
        shape=(seq_len, max_head_dim),
        strides=(qk_stride_seq, qk_stride_d),
        offsets=(query_tile_idx * tile_size_q, 0,),
        block_shape=(tile_size_q, max_head_dim),
        order=(1, 0)
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + batch * qk_stride_batch,
        shape=(seq_len, max_head_dim),
        strides=(qk_stride_seq, qk_stride_d),
        offsets=(0, 0),
        block_shape=(tile_size_kv, max_head_dim),
        order=(1, 0)
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + batch * v_stride_batch,
        shape=(seq_len, max_head_dim),
        strides=(v_stride_seq, v_stride_d),
        offsets=(0, 0),
        block_shape=(tile_size_kv, max_head_dim),
        order=(1, 0)
    )

    o_block_ptr = tl.make_block_ptr(
        o_ptr + batch * o_stride_batch,
        shape=(seq_len, max_head_dim),
        strides=(o_stride_seq, o_stride_d),
        offsets=(query_tile_idx * tile_size_q, 0,),
        block_shape=(tile_size_q, max_head_dim),
        order=(1, 0)
    )

    l_block_ptr = tl.make_block_ptr(
        l_ptr + batch*l_stride_batch,
        shape=(seq_len, 1),
        strides=(l_stride_seq, 1),
        offsets=(query_tile_idx *tile_size_q, 0),
        block_shape=(tile_size_q, 1),
        order=(1,0)
    )

    Qi = tl.load(q_block_ptr, boundary_check=(1,), padding_option='zero')

    m = tl.full(shape=(tile_size_q, 1), value=float('-inf'), dtype=tl.float32)
    l = tl.zeros((tile_size_q, 1), dtype=tl.float32)
    Oi = tl.zeros((tile_size_q, max_head_dim), dtype=tl.float32)

    for j in range(tl.cdiv(seq_len, tile_size_kv)):
        # Load KV
        Kj = tl.load(k_block_ptr, boundary_check=(1,), padding_option='zero')
        Vj = tl.load(v_block_ptr, boundary_check=(1,), padding_option='zero')

        # computeattention
        # Sij = tl.dot(Qi, Kj.T) * scale
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale

        # Compute row maximum
        m_cur = tl.max(Sij, axis=-1, return_indices=False).reshape(tile_size_q, 1)
        m_new = tl.maximum(m, m_cur)

        Pij = tl.exp(Sij - m_new)

        # update row-sum
        l_cur = tl.sum(Pij, axis=-1).reshape(tile_size_q, 1)
        l_new = tl.exp(m - m_new)*l + l_cur

        PV = tl.dot(Pij.to(Vj.dtype), Vj)

        # Update O
        Oi = tl.exp(m - m_new) * Oi + PV

        m = m_new
        l = l_new

        # Advance KV
        k_block_ptr = k_block_ptr.advance((tile_size_kv, 0))
        v_block_ptr = v_block_ptr.advance((tile_size_kv, 0))

    Oi /= l
    Oi = Oi.to(tl.float32)
    tl.store(o_block_ptr, Oi, boundary_check=(1,))

    logsuml = m + tl.log(l)
    tl.store(l_block_ptr, logsuml, boundary_check=(1,))


class FlashAttentionTorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys    d_k"],
        V: Float[Tensor, " ... keys    d_v"],
        causal: bool = False
    ) -> Float[Tensor, " ... queries d_v"]:
        if len(Q.shape) > 3:
            Q = rearrange(Q, "... seq d_k -> (...) seq d_k")
            K = rearrange(Q, "... seq d_k -> (...) seq d_k")
            V = rearrange(Q, "... seq d_v -> (...) seq d_v")

        O, L = flash_attn_forward(Q, K, V, causal)
        L = rearrange(L, "batch seq 1 -> batch seq")

        ctx.save_for_backward(O, L)
        return O


    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("Backward pass not implemented")

class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys    d_k"],
        V: Float[Tensor, " ... keys    d_v"],
        causal: bool = False
    ) -> Float[Tensor, " ... queries d_v"]:
        rearrange_back = False
        if len(Q.shape) > 3:
            Q = rearrange(Q, "... seq d_k -> (...) seq d_k")
            K = rearrange(Q, "... seq d_k -> (...) seq d_k")
            V = rearrange(Q, "... seq d_v -> (...) seq d_v")

        batch_dim = Q.shape[0]
        seq_len = Q.shape[1]
        d_k = Q.shape[2]
        d_v = V.shape[2]

        ctx.tile_size_q = 32
        ctx.tile_size_kv = 32
        ctx.max_head_dim = d_k

        grid = (batch_dim, triton.cdiv(seq_len, ctx.tile_size_q))
        O = torch.zeros((batch_dim, seq_len, d_v), dtype=torch.float32, device=Q.device)
        L = torch.zeros((batch_dim, seq_len, 1), dtype=torch.float32, device=Q.device)
        flash_attn_forward_triton[grid](
            Q, K, V, O, L,
            batch_dim, seq_len,
            Q.stride(0), Q.stride(1), Q.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            tile_size_q = ctx.tile_size_q,
            tile_size_kv = ctx.tile_size_kv,
            max_head_dim = ctx.max_head_dim
        )
        # print(O[0, :, :])
        # pytest.exit(0)
        L = rearrange(L, "batch seq 1 -> batch seq")
        ctx.save_for_backward(O, L)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("Backward pass not implemented")
