import torch
from torch import Tensor
from jaxtyping import Float, Bool, Int
import triton
import triton.language as tl
from einops import rearrange, einsum
import pytest
from triton.language.core import dtype

def cdiv(x: int, y: int):
    return (x + y - 1) // y

def flash_attn_forward(
    Q: Float[Tensor, "batch queries d_k"],
    K: Float[Tensor, "batch keys    d_k"],
    V: Float[Tensor, "batch keys    d_v"],
    causal: bool = False,
    tileQ: int = 8,
    tileKV: int = 16,
) -> tuple[Float[Tensor, " ... queries d_v"], Float[Tensor, "... queries"]]:
    assert len(Q.shape) == 3
    assert len(K.shape) == 3
    assert len(V.shape) == 3

    batch_dim = Q.shape[0]
    num_queries = Q.shape[1]
    num_keys = K.shape[1]
    d_k = Q.shape[2]
    d_v = V.shape[2]

    softmax_scale = 1. / (d_k**0.5)
    O = torch.zeros((batch_dim, num_queries, d_v), dtype=torch.float32, device=Q.device)
    L = torch.zeros((batch_dim, num_queries, 1), dtype=torch.float32, device=Q.device)

    for b in range(batch_dim):
        for q_chunk in range(cdiv(num_queries, tileQ)):
            startI = q_chunk * tileQ
            endI = min((q_chunk+1)*tileQ, num_queries)
            sliceI = (b, slice(startI, endI), slice(None))
            m = torch.full((tileQ, 1), fill_value=-torch.inf, dtype=torch.float32, device=Q.device)
            l = torch.zeros((tileQ, 1), dtype=torch.float32, device=Q.device)
            Qi = Q[sliceI]
            for kv_chunk in range(cdiv(num_keys, tileKV)):
                # Load K,V
                startJ = kv_chunk * tileKV
                endJ = min((kv_chunk+1)*tileKV, num_keys)
                sliceJ = (b, slice(startJ, endJ), slice(None))
                Kj = K[sliceJ]
                Vj = V[sliceJ]
                # Compute QK
                Sij = einsum(Qi, Kj, "query d_k, key d_k -> query key") * softmax_scale
                # Compute new row-max
                m_cur = torch.max(Sij, dim=-1)[0].reshape(tileQ, 1).to()
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
            L[b, startI:endI] = m + torch.log(l)

    return O, L

def flash_attn_backward(Q, K, V, O, dO, L, tileQ, tileKV):
    batch_dim = Q.shape[0]
    num_queries = Q.shape[1]
    assert(batch_dim == K.shape[0] == V.shape[0] == O.shape[0] == dO.shape[0] == L.shape[0])
    num_keys = V.shape[1]
    d_k = Q.shape[2]
    d_v = V.shape[2]

    dQ = torch.zeros_like(Q, device=Q.device)
    dV = torch.zeros_like(V, device=V.device)
    dK = torch.zeros_like(K, device=K.device)
    softmax_scale = d_k**(-0.5)

    D = torch.empty((batch_dim, num_queries), device=O.device)
    for b in range(batch_dim):
        for q_start in range(0, num_queries, tileQ):
            sliceI = slice(q_start, min(q_start + tileQ, num_queries))
            OdOi = O[b, sliceI, :] * dO[b, sliceI, :]
            D[b, sliceI] = OdOi.sum(axis=-1)

    for b in range(batch_dim):
        for kv_start in range(0, num_keys, tileKV):
            sliceJ = slice(kv_start, min(kv_start+tileKV, num_keys))
            Kj = K[b, sliceJ, :]
            Vj = V[b, sliceJ, :]

            dKj = torch.zeros(tileKV, d_k, device=Q.device)
            dVj = torch.zeros(tileKV, d_v, device=Q.device)

            for q_start in range(0, num_queries, tileQ):
                # update dVj
                sliceI = slice(q_start, min(q_start + tileQ, num_queries))
                Qi = Q[b, sliceI, :] # tQ dk
                Sij = einsum(Qi, Kj, "query d_k, key d_k -> query key") * softmax_scale # (tQ x tK)
                Li = L[b, sliceI].reshape(tileQ, 1)
                Pij = torch.exp(Sij - Li) # (tQ x tK)

                dOi = dO[b, sliceI, :] # tQ, d_v

                # dVj += einsum(Pij, dOi, "query key, query d_v -> key d_v") # (k, d_v)
                dVj += Pij.T @ dOi

                # update dKj
                dPij = dOi @ Vj.T
                Di = D[b, sliceI].reshape(tileQ, 1)
                dSij = Pij * (dPij - Di) * softmax_scale
                dKj += dSij.T @ Qi

                # update dQi += dSij @ Kj; must be atomic!!!!!!
                dQ[b, sliceI, :] += einsum(dSij, Kj, "q k, k d_k -> q d_k")

            dV[b, sliceJ, :] = dVj
            dK[b, sliceJ, :] = dKj

    return dQ, dK, dV

@triton.autotune(configs=[
    triton.Config(kwargs={'tileQ': 64, 'tileKV': 32}, num_warps=2),
    triton.Config(kwargs={'tileQ': 64, 'tileKV': 32}, num_warps=4),
    triton.Config(kwargs={'tileQ': 64, 'tileKV': 32}, num_warps=8),
    triton.Config(kwargs={'tileQ': 128, 'tileKV': 64}, num_warps=8),
    ],
    key=['num_queries'] # evaluated anytime the value of num_queqries changes
)
@triton.jit
def flash_attn_forward_triton(
    q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
    num_queries, num_keys, d_k, d_v,
    q_stride_batch, q_stride_seq, q_stride_d,
    k_stride_batch, k_stride_seq, k_stride_d,
    v_stride_batch, v_stride_seq, v_stride_d,
    o_stride_batch, o_stride_seq, o_stride_d,
    l_stride_batch, l_stride_seq,
    tileQ: tl.constexpr,
    tileKV: tl.constexpr,
    maxD: tl.constexpr,
    is_causal: tl.constexpr
):
    batch = tl.program_id(1) # blockIdx.y
    query_tile_idx = tl.program_id(0)
    scale = maxD**(-0.5)
    q_start = query_tile_idx * tileQ

    q_block_ptr = tl.make_block_ptr(
        q_ptr + batch * q_stride_batch,
        shape=(num_queries, d_k),
        strides=(q_stride_seq, q_stride_d),
        offsets=(q_start, 0,),
        block_shape=(tileQ, maxD),
        order=(1, 0)
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + batch * k_stride_batch,
        shape=(num_keys, d_k),
        strides=(k_stride_seq, k_stride_d),
        offsets=(0, 0),
        block_shape=(tileKV, maxD),
        order=(1, 0)
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + batch * v_stride_batch,
        shape=(num_keys, d_v),
        strides=(v_stride_seq, v_stride_d),
        offsets=(0, 0),
        block_shape=(tileKV, maxD),
        order=(1, 0)
    )

    o_block_ptr = tl.make_block_ptr(
        o_ptr + batch * o_stride_batch,
        shape=(num_queries, d_v),
        strides=(o_stride_seq, o_stride_d),
        offsets=(q_start, 0,),
        block_shape=(tileQ, maxD),
        order=(1, 0)
    )

    l_block_ptr = tl.make_block_ptr(
        l_ptr + batch*l_stride_batch,
        shape=(num_keys, 1),
        strides=(l_stride_seq, 1),
        offsets=(q_start, 0),
        block_shape=(tileQ, 1),
        order=(1,0)
    )

    Qi = tl.load(q_block_ptr, boundary_check=(1,), padding_option='zero')

    m = tl.full(shape=(tileQ, 1), value=float('-inf'), dtype=tl.float32)
    l = tl.zeros((tileQ, 1), dtype=tl.float32)
    Oi = tl.zeros((tileQ, maxD), dtype=tl.float32)

    row_indices = q_start + tl.arange(0, tileQ)

    for j in range(tl.cdiv(num_keys, tileKV)):
        # Load KV
        Kj = tl.load(k_block_ptr, boundary_check=(1,), padding_option='zero')
        Vj = tl.load(v_block_ptr, boundary_check=(1,), padding_option='zero')

        k_start = j * tileKV
        k_end = (j+1) * tileKV
        col_indices = j*tileKV + tl.arange(0, tileKV)

        # compute scores
        Sij = tl.dot(Qi, Kj.T) * scale

        if is_causal:
            Sij = tl.where(row_indices[:,None] >= col_indices[None,:], Sij, float("-inf"))

        # Compute row maximum
        m_cur = tl.max(Sij, axis=-1, return_indices=False).reshape(tileQ, 1)
        m_new = tl.maximum(m, m_cur)

        Pij = tl.exp(Sij - m_new)

        # update row-sum
        l_cur = tl.sum(Pij, axis=-1).reshape(tileQ, 1)
        l_new = tl.exp(m - m_new)*l + l_cur

        PV = tl.dot(Pij.to(Vj.dtype), Vj)

        # Update weights
        Oi = tl.exp(m - m_new) * Oi + PV

        m = m_new
        l = l_new

        # Advance KV
        k_block_ptr = k_block_ptr.advance((tileKV, 0))
        v_block_ptr = v_block_ptr.advance((tileKV, 0))

    Oi /= l
    Oi = Oi.to(tl.float32)
    tl.store(o_block_ptr, Oi, boundary_check=(1,))

    logsuml = m + tl.log(l)
    tl.store(l_block_ptr, logsuml, boundary_check=(1,))

@triton.jit
def compute_D_matrix(o_ptr, do_ptr, d_ptr,
    odo_stride_batch, odo_stride_seq, odo_stride_d,
    d_stride_batch, d_stride_seq,
    num_queries, d_v,
    tile_size_seq: tl.constexpr, tile_size_d: tl.constexpr
):
    batch = tl.program_id(1)
    query_tile_id = tl.program_id(0)
    o_start = query_tile_id * tile_size_seq

    o_block_ptr = tl.make_block_ptr(
        o_ptr + batch * odo_stride_batch,
        shape=(num_queries, d_v),
        strides=(odo_stride_seq, odo_stride_d),
        offsets=(o_start, 0,),
        block_shape=(tile_size_seq, tile_size_d),
        order=(1, 0)
    )

    do_block_ptr = tl.make_block_ptr(
        do_ptr + batch * odo_stride_batch,
        shape=(num_queries, d_v),
        strides=(odo_stride_seq, odo_stride_d),
        offsets=(o_start, 0,),
        block_shape=(tile_size_seq, tile_size_d),
        order=(1, 0)
    )

    d_block_ptr = tl.make_block_ptr(
        d_ptr + batch * d_stride_batch,
        shape=(num_queries, 1),
        strides=(d_stride_seq, 1),
        offsets=(o_start, 0),
        block_shape=(tile_size_seq, 1),
        order=(1, 0)
    )

    Dij = tl.zeros((tile_size_seq, tile_size_d), dtype=tl.float32)
    for j in range(tl.cdiv(d_v, tile_size_d)):
        Oi = tl.load(o_block_ptr, boundary_check=(1,), padding_option='zero')
        dOi = tl.load(do_block_ptr, boundary_check=(1,), padding_option='zero')
        Dij += Oi*dOi
        o_block_ptr = o_block_ptr.advance((0, tile_size_d))
        do_block_ptr = do_block_ptr.advance((0, tile_size_d))
    Di = tl.sum(Dij, axis=-1).reshape(tile_size_seq, 1)
    tl.store(d_block_ptr, Di, boundary_check=(1,))


@triton.jit
def flash_attn_backward_triton(
    q_ptr, k_ptr, v_ptr, o_ptr,
    do_ptr,
    l_ptr, d_ptr,
    dq_ptr, dk_ptr, dv_ptr,
    num_queries, num_keys, d_k, d_v,
    q_stride_batch, q_stride_seq, q_stride_d,
    k_stride_batch, k_stride_seq, k_stride_d,
    v_stride_batch, v_stride_seq, v_stride_d,
    o_stride_batch, o_stride_seq, o_stride_d,
    dl_stride_batch, dl_stride_seq,
    tileQ: tl.constexpr, tileKV: tl.constexpr, maxD: tl.constexpr,
    is_causal: tl.constexpr,
):
    batch = tl.program_id(1) # blockIdx.y
    kv_tile_idx = tl.program_id(0)  # blockidx.x
    kv_start = kv_tile_idx * tileKV
    scale = maxD**(-0.5)

    k_block_ptr = tl.make_block_ptr(
        k_ptr + batch * k_stride_batch,
        shape=(num_keys, d_k),
        strides=(k_stride_seq, k_stride_d),
        offsets=(kv_start, 0),
        block_shape=(tileKV, maxD),
        order=(1, 0)
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + batch * v_stride_batch,
        shape=(num_keys, d_v),
        strides=(v_stride_seq, v_stride_d),
        offsets=(kv_start, 0),
        block_shape=(tileKV, maxD),
        order=(1, 0)
    )


    q_block_ptr = tl.make_block_ptr(
        q_ptr + batch * q_stride_batch,
        shape=(num_queries, d_k),
        strides=(q_stride_seq, q_stride_d),
        offsets=(0, 0,),
        block_shape=(tileQ, maxD),
        order=(1, 0)
    )

    do_block_ptr = tl.make_block_ptr(
        do_ptr + batch * o_stride_batch,
        shape=(num_queries, d_v),
        strides=(o_stride_seq, o_stride_d),
        offsets=(0, 0),
        block_shape=(tileQ, maxD),
        order=(1, 0)
    )

    l_block_ptr = tl.make_block_ptr(
        l_ptr + batch*dl_stride_batch,
        shape=(num_queries, 1),
        strides=(dl_stride_seq, 1),
        offsets=(0, 0),
        block_shape=(tileQ, 1),
        order=(1,0)
    )

    d_block_ptr = tl.make_block_ptr(
        d_ptr + batch*dl_stride_batch,
        shape=(num_queries, 1),
        strides=(dl_stride_seq, 1),
        offsets=(0, 0),
        block_shape=(tileQ, 1),
        order=(1,0)
    )

    Kj = tl.load(k_block_ptr, boundary_check=(1,), padding_option='zero')
    Vj = tl.load(v_block_ptr, boundary_check=(1,), padding_option='zero')
    dKj = tl.zeros((tileKV, maxD), dtype=tl.float32)
    dVj = tl.zeros((tileKV, maxD), dtype=tl.float32)

    col_indices = kv_start + tl.arange(0, tileKV)

    for q_start in range(0, num_queries, tileQ):
        row_indices = q_start + tl.arange(0, tileQ)
        Qi = tl.load(q_block_ptr, boundary_check=(1,), padding_option='zero')   # (tQ x d_k)
        Li = tl.load(l_block_ptr, boundary_check=(1,), padding_option='zero')   # (tQ x 1)
        dOi = tl.load(do_block_ptr, boundary_check=(1,), padding_option='zero') # (tQ x d_v)

        # update dVj
        Sij = tl.dot(Qi, Kj.T) * scale # (tQ x tK)
        if is_causal:
            Sij = tl.where(row_indices[:,None] >= col_indices[None,:], Sij, float("-inf"))
        Pij = tl.exp(Sij - Li)  # (tQ x tKV)
        dVj += tl.dot(Pij.T, dOi)  # (tKV x d_v) = (tKV x tQ) @ (tQ x d_v)

        # update dKj
        dPij = tl.dot(dOi, Vj.T)
        Di = tl.load(d_block_ptr, boundary_check=(1,),).reshape(tileQ, 1)
        dSij = Pij * (dPij - Di) * scale
        dKj += tl.dot(dSij.T, Qi)

        delta_dQi = tl.dot(dSij, Kj)

        # Atomically update dQ
        q_row_idx = q_start + tl.arange(0, tileQ)[:, None]  # Current query rows
        q_col_idx = tl.arange(0, maxD)[None, :]             # All feature dimensions
        q_mask = (q_row_idx < num_queries) & (q_col_idx < maxD)
        dq_ptrs = dq_ptr + batch * q_stride_batch + q_row_idx * q_stride_seq + q_col_idx * q_stride_d
        tl.atomic_add(dq_ptrs, delta_dQi, mask=q_mask)

        q_block_ptr = q_block_ptr.advance((tileQ, 0))
        l_block_ptr = l_block_ptr.advance((tileQ, 0))
        d_block_ptr = d_block_ptr.advance((tileQ, 0))
        do_block_ptr = do_block_ptr.advance((tileQ, 0))

    # store dK, dV
    dk_block_ptr = tl.make_block_ptr(
        dk_ptr + batch * q_stride_batch,
        shape=(num_keys, d_k),
        strides=(q_stride_seq, q_stride_d),
        offsets=(kv_start, 0),
        block_shape=(tileKV, maxD),
        order=(1, 0)
    )

    dv_block_ptr = tl.make_block_ptr(
        dv_ptr + batch * v_stride_batch,
        shape=(num_keys, d_v),
        strides=(v_stride_seq, v_stride_d),
        offsets=(kv_start, 0),
        block_shape=(tileKV, maxD),
        order=(1, 0)
    )

    tl.store(dk_block_ptr, dKj, boundary_check=(1,))
    tl.store(dv_block_ptr, dVj, boundary_check=(1,))



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
            K = rearrange(K, "... seq d_k -> (...) seq d_k")
            V = rearrange(V, "... seq d_v -> (...) seq d_v")

        num_queries = Q.shape[1]
        num_keys = K.shape[1]
        # ctx.tileQ = min(8, num_queries)
        # ctx.tileKV = min(16, num_keys)
        ctx.tileQ = min(2**13, num_queries)
        ctx.tileKV = min(2**13, num_keys)

        O, L = flash_attn_forward(Q, K, V, causal, ctx.tileQ, ctx.tileKV)
        L = rearrange(L, "batch seq 1 -> batch seq")

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = causal
        return O


    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors

        dQ, dK, dV = flash_attn_backward(Q, K, V, O, dO, L, ctx.tileQ, ctx.tileKV)
        dCausal = None
        return dQ, dK, dV, dCausal

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
        num_queries = Q.shape[1]
        num_keys = K.shape[1]
        d_k = Q.shape[2]
        d_v = V.shape[2]

        ctx.tile_size_q = 16
        ctx.tile_size_kv = 16
        ctx.max_head_dim = d_k
        ctx.is_causal = causal

        grid = (triton.cdiv(num_queries, ctx.tile_size_q), batch_dim)
        O = torch.zeros((batch_dim, num_queries, d_v), dtype=torch.float32, device=Q.device)
        L = torch.zeros((batch_dim, num_queries, 1), dtype=torch.float32, device=Q.device)
        flash_attn_forward_triton[grid](
            Q, K, V, O, L,
            num_queries, num_keys, d_k, d_v,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            # tileQ = ctx.tile_size_q,
            # tileKV = ctx.tile_size_kv,
            maxD = ctx.max_head_dim,
            is_causal=ctx.is_causal
        )
        L = rearrange(L, "batch seq 1 -> batch seq")
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = causal
        return O

    @staticmethod
    def backward(ctx, dO: Float[Tensor, "... queries d_v"]
    ) -> tuple[Float[Tensor, "... queries d_k"], # dQ
               Float[Tensor, "... keys d_k"],    # dK
               Float[Tensor, "... values d_v"],  # dV
               None]: # None as is_causal is not differentiable
        Q, K, V, O, L = ctx.saved_tensors
        # print(f"\nctx.is_causal = {ctx.is_causal}")

        dQ = torch.zeros_like(Q, device=Q.device)
        dK = torch.empty_like(K, device=K.device)
        dV = torch.empty_like(V, device=V.device)
        dCausal = None

        batch_dim = Q.shape[0]
        num_queries = Q.shape[1]
        num_keys = K.shape[1]
        d_k = Q.shape[2]
        d_v = V.shape[2]

        # D = torch.empty((batch_dim, seq_len), device=O.device)
        D = torch.empty((batch_dim, num_queries), device=O.device)
        ctx.tile_size_seq = 4
        ctx.tile_size_d = 1*32
        grid = (triton.cdiv(num_queries, ctx.tile_size_seq), batch_dim)
        compute_D_matrix[grid](
            O, dO, D,
            O.stride(0), O.stride(1), O.stride(2),
            D.stride(0), D.stride(1),
            num_queries, d_v,
            tile_size_seq=ctx.tile_size_seq,
            tile_size_d=ctx.tile_size_d,
        )

        grid = (triton.cdiv(num_keys, ctx.tile_size_kv), batch_dim)
        flash_attn_backward_triton[grid] (
            Q, K, V, O,
            dO,
            L, D,
            dQ, dK, dV,
            num_queries, num_keys, d_k, d_v,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            D.stride(0), D.stride(1),
            tileQ=ctx.tile_size_q,
            tileKV=ctx.tile_size_kv,
            maxD=ctx.max_head_dim,
            is_causal=ctx.is_causal
        )

        return dQ, dK, dV, dCausal
