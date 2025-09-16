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
    tileQ = 16
    tileK = 16

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

def flash_attn_backward(Q, K, V, O, dO, L):
    batch_dim = Q.shape[0]
    seq_len = Q.shape[1]
    d_k = Q.shape[2]
    d_v = V.shape[2]
    dQ = torch.zeros_like(Q, device=Q.device)
    dV = torch.zeros_like(V, device=V.device)
    dK = torch.zeros_like(K, device=K.device)
    softmax_scale = d_k**(-0.5)

    tileQ = 16
    tileK = 16
    # D = einsum(O, dO, "... query d_v, ... query d_v -> ... query d_v").sum(axis=-1)
    D = torch.empty((batch_dim, seq_len), device=O.device)
    for b in range(batch_dim):
        for q_start in range(0, seq_len, tileK):
            sliceI = slice(q_start, min(q_start + tileQ, seq_len))
            OdOi = O[b, sliceI, :] * dO[b, sliceI, :]
            D[b, sliceI] = OdOi.sum(axis=-1)

    for b in range(batch_dim):
        for kv_start in range(0, seq_len, tileK):
            sliceJ = slice(kv_start, min(kv_start+tileK, seq_len))
            Kj = K[b, sliceJ, :]
            Vj = V[b, sliceJ, :]
            dVj = torch.zeros(tileK, d_v)
            dKj = torch.zeros(tileK, d_k)
            for q_start in range(0, seq_len, tileQ):
                sliceI = slice(q_start, min(q_start + tileQ, seq_len))
                Qi = Q[b, sliceI, :] # tQ dk
                Sij = einsum(Qi, Kj, "query d_k, key d_k -> query key") * softmax_scale # (tQ x tK)

                Li = L[b, sliceI]
                Pij = torch.exp(Sij - Li[:, None]) # (tQ x tK)

                dOi = dO[b, sliceI, :] # tQ, d_v

                # dVj += einsum(Pij, dOi, "query key, query d_v -> key d_v") # (k, d_v)
                dVj += Pij.T @ dOi

                dPij = einsum(dOi, Vj, "q d_v, k d_v -> q k") # (q, k)

                Di = D[b, sliceI]
                dSij = Pij * (dPij - Di[:, None]) * softmax_scale

                # update dQi += dSij @ Kj; must be atomic!!!!!!
                dQ[b, sliceI, :] += einsum(dSij, Kj, "q k, k d_k -> q d_k")

                dKj += einsum(dSij, Qi, "q k, q d_k -> k d_k")

            dV[b, sliceJ, :] = dVj
            dK[b, sliceJ, :] = dKj
            # print(f"dV = {dV}")

    return dQ, dK, dV

@triton.jit
def flash_attn_forward_triton(
    q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
    batch_dim, seq_len,
    qk_stride_batch, qk_stride_seq, qk_stride_d,
    v_stride_batch, v_stride_seq, v_stride_d,
    o_stride_batch, o_stride_seq, o_stride_d,
    l_stride_batch, l_stride_seq,
    tile_size_q: tl.constexpr, tile_size_kv: tl.constexpr,
    max_head_dim: tl.constexpr,
    is_causal: tl.constexpr
):
    batch = tl.program_id(1) # blockIdx.y
    query_tile_idx = tl.program_id(0)
    scale = 1. / max_head_dim**0.5
    q_start = query_tile_idx * tile_size_q

    q_block_ptr = tl.make_block_ptr(
        q_ptr + batch * qk_stride_batch,
        shape=(seq_len, max_head_dim),
        strides=(qk_stride_seq, qk_stride_d),
        offsets=(q_start, 0,),
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
        offsets=(q_start, 0,),
        block_shape=(tile_size_q, max_head_dim),
        order=(1, 0)
    )

    l_block_ptr = tl.make_block_ptr(
        l_ptr + batch*l_stride_batch,
        shape=(seq_len, 1),
        strides=(l_stride_seq, 1),
        offsets=(q_start, 0),
        block_shape=(tile_size_q, 1),
        order=(1,0)
    )

    Qi = tl.load(q_block_ptr, boundary_check=(1,), padding_option='zero')

    m = tl.full(shape=(tile_size_q, 1), value=float('-inf'), dtype=tl.float32)
    l = tl.zeros((tile_size_q, 1), dtype=tl.float32)
    Oi = tl.zeros((tile_size_q, max_head_dim), dtype=tl.float32)

    row_indices = q_start + tl.arange(0, tile_size_q)

    for j in range(tl.cdiv(seq_len, tile_size_kv)):
        # Load KV
        Kj = tl.load(k_block_ptr, boundary_check=(1,), padding_option='zero')
        Vj = tl.load(v_block_ptr, boundary_check=(1,), padding_option='zero')

        k_start = j * tile_size_kv
        k_end = (j+1) * tile_size_kv
        col_indices = j*tile_size_kv + tl.arange(0, tile_size_kv)

        # compute scores
        Sij = tl.dot(Qi, Kj.T) * scale

        if is_causal:
            Sij = tl.where(row_indices[:,None] >= col_indices[None,:], Sij, float("-inf"))


        # Compute row maximum
        m_cur = tl.max(Sij, axis=-1, return_indices=False).reshape(tile_size_q, 1)
        m_new = tl.maximum(m, m_cur)

        Pij = tl.exp(Sij - m_new)

        # update row-sum
        l_cur = tl.sum(Pij, axis=-1).reshape(tile_size_q, 1)
        l_new = tl.exp(m - m_new)*l + l_cur

        PV = tl.dot(Pij.to(Vj.dtype), Vj)

        # Update weights
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

@triton.jit
def compute_D_matrix(o_ptr, do_ptr, d_ptr,
    odo_stride_batch, odo_stride_seq, odo_stride_d,
    d_stride_batch, d_stride_seq,
    seq_len, d_v,
    tile_size_seq: tl.constexpr, tile_size_d: tl.constexpr
):
    batch = tl.program_id(1)
    query_tile_id = tl.program_id(0)
    o_start = query_tile_id * tile_size_seq

    o_block_ptr = tl.make_block_ptr(
        o_ptr + batch * odo_stride_batch,
        shape=(seq_len, d_v),
        strides=(odo_stride_seq, odo_stride_d),
        offsets=(o_start, 0,),
        block_shape=(tile_size_seq, tile_size_d),
        order=(1, 0)
    )

    do_block_ptr = tl.make_block_ptr(
        do_ptr + batch * odo_stride_batch,
        shape=(seq_len, d_v),
        strides=(odo_stride_seq, odo_stride_d),
        offsets=(o_start, 0,),
        block_shape=(tile_size_seq, tile_size_d),
        order=(1, 0)
    )

    d_block_ptr = tl.make_block_ptr(
        d_ptr + batch * d_stride_batch,
        shape=(seq_len, 1),
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
    seq_len,
    qk_stride_batch, qk_stride_seq, qk_stride_d,
    v_stride_batch, v_stride_seq, v_stride_d,
    o_stride_batch, o_stride_seq, o_stride_d,
    dl_stride_batch, dl_stride_seq, dl_stride_d,
    tileQ: tl.constexpr, tileKV: tl.constexpr, maxD: tl.constexpr,
    is_causal: tl.constexpr,
):
    batch = tl.program_id(1) # blockIdx.y
    # query_tile_idx = tl.program_id(0)
    # q_start = query_tile_idx * tileQ
    kv_tile_idx = tl.program_id(0)
    kv_start = kv_tile_idx * tileKV
    scale = 1. / maxD**0.5

    k_block_ptr = tl.make_block_ptr(
        k_ptr + batch * qk_stride_batch,
        shape=(seq_len, maxD),
        strides=(qk_stride_seq, qk_stride_d),
        offsets=(kv_start, 0),
        block_shape=(tileKV, maxD),
        order=(1, 0)
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + batch * v_stride_batch,
        shape=(seq_len, maxD),
        strides=(v_stride_seq, v_stride_d),
        offsets=(kv_start, 0),
        block_shape=(tileKV, maxD),
        order=(1, 0)
    )


    dk_block_ptr = tl.make_block_ptr(
        dk_ptr + batch * qk_stride_batch,
        shape=(seq_len, maxD),
        strides=(qk_stride_seq, qk_stride_d),
        offsets=(kv_start, 0),
        block_shape=(tileKV, maxD),
        order=(1, 0)
    )

    dv_block_ptr = tl.make_block_ptr(
        dv_ptr + batch * v_stride_batch,
        shape=(seq_len, maxD),
        strides=(v_stride_seq, v_stride_d),
        offsets=(kv_start, 0),
        block_shape=(tileKV, maxD),
        order=(1, 0)
    )

    Kj = tl.load(k_block_ptr, boundary_check=(1,), padding_option='zero')
    Vj = tl.load(v_block_ptr, boundary_check=(1,), padding_option='zero')
    dKj = tl.zeros((tileKV, maxD), dtype=Kj.dtype)
    dVj = tl.zeros((tileKV, maxD), dtype=Vj.dtype)
    # tl.device_print(f"dKj.shape = {dKj.shape[0]}, {dKj.shape[1]}")

    q_block_ptr = tl.make_block_ptr(
        q_ptr + batch * qk_stride_batch,
        shape=(seq_len, maxD),
        strides=(qk_stride_seq, qk_stride_d),
        offsets=(0, 0,),
        block_shape=(tileQ, maxD),
        order=(1, 0)
    )

    # dq_block_ptr = tl.make_block_ptr(
    #     dq_ptr + batch * qk_stride_batch,
    #     shape=(seq_len, maxD),
    #     strides=(qk_stride_seq, qk_stride_d),
    #     offsets=(0, 0,),
    #     block_shape=(tileQ, maxD),
    #     order=(1, 0)
    # )

    do_block_ptr = tl.make_block_ptr(
        do_ptr + batch * o_stride_batch,
        shape=(seq_len, maxD),
        strides=(o_stride_seq, o_stride_d),
        offsets=(0, 0),
        block_shape=(tileQ, maxD),
        order=(1, 0)
    )

    l_block_ptr = tl.make_block_ptr(
        l_ptr + batch*dl_stride_batch,
        shape=(seq_len, 1),
        strides=(dl_stride_seq, 1),
        offsets=(0, 0),
        block_shape=(tileQ, 1),
        order=(1,0)
    )

    d_block_ptr = tl.make_block_ptr(
        d_ptr + batch*dl_stride_batch,
        shape=(seq_len, 1),
        strides=(dl_stride_seq, 1),
        offsets=(0, 0),
        block_shape=(tileQ, 1),
        order=(1,0)
    )

    for q_start in range(0, seq_len, tileQ):
        Qi = tl.load(q_block_ptr, boundary_check=(1,), padding_option='zero')
        Sij = tl.dot(Qi, Kj.T) * scale # (tQ x tK)
        # tl.device_print(f"Sij.shape = {Sij.shape[0]}, {Sij.shape[1]}") # 32 x 32

        Li = tl.load(l_block_ptr, boundary_check=(1,), ).reshape(tileQ, 1)
        # Pij = tl.exp(Sij - Li[:, None])  # (tQ x tKV)
        Pij = tl.exp(Sij - Li)  # (tQ x tKV)

        dOi = tl.load(do_block_ptr, boundary_check=(1,), padding_option='zero') # (tQ x d_v)

        dVj += tl.dot(Pij.T, dOi)  # (tKV x d_v) = (tKV x tQ) @ (tQ x d_v)

        dPij = tl.dot(dOi, Vj.T)

        Di = tl.load(d_block_ptr, boundary_check=(1,),).reshape(tileQ, 1)
        dSij = Pij * (dPij - Di) * scale

        # dQi = tl.load(q_block_ptr, boundary_check=(1,), padding_option='zero')
        delta_dQi = tl.dot(dSij, Kj)

        # Create 2D pointer array for atomic add
        q_row_idx = q_start + tl.arange(0, tileQ)[:, None]  # Current query rows
        q_col_idx = tl.arange(0, maxD)[None, :]             # All feature dimensions
        q_mask = (q_row_idx < seq_len) & (q_col_idx < maxD)

        dq_ptrs = dq_ptr + batch * qk_stride_batch + q_row_idx * qk_stride_seq + q_col_idx * qk_stride_d
        tl.atomic_add(dq_ptrs, delta_dQi, mask=q_mask)

        dKj += tl.dot(dSij.T, Qi)

        q_block_ptr = q_block_ptr.advance((tileQ, 0))
        l_block_ptr = l_block_ptr.advance((tileQ, 0))
        d_block_ptr = d_block_ptr.advance((tileQ, 0))
        # dq_block_ptr = dq_block_ptr.advance((tileQ, 0))
        do_block_ptr = do_block_ptr.advance((tileQ, 0))

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
            K = rearrange(Q, "... seq d_k -> (...) seq d_k")
            V = rearrange(Q, "... seq d_v -> (...) seq d_v")

        O, L = flash_attn_forward(Q, K, V, causal)
        L = rearrange(L, "batch seq 1 -> batch seq")

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = causal
        return O


    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors

        dQ, dK, dV = flash_attn_backward(Q, K, V, O, dO, L)
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
        seq_len = Q.shape[1]
        d_k = Q.shape[2]
        d_v = V.shape[2]

        ctx.tile_size_q = 32
        ctx.tile_size_kv = 32
        ctx.max_head_dim = d_k
        ctx.is_causal = causal

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
            max_head_dim = ctx.max_head_dim,
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
        print(f"\nctx.is_causal = {ctx.is_causal}")

        dQ = torch.zeros_like(Q, device=Q.device)
        dK = torch.empty_like(K, device=K.device)
        dV = torch.empty_like(V, device=V.device)
        dCausal = None

        batch_dim = Q.shape[0]
        seq_len = Q.shape[1]
        d_k = Q.shape[2]
        d_v = V.shape[2]

        # D = torch.empty((batch_dim, seq_len), device=O.device)
        D = torch.empty((batch_dim, seq_len), device=O.device)
        ctx.tile_size_seq = 4
        ctx.tile_size_d = 1*32
        grid = (batch_dim, triton.cdiv(seq_len, ctx.tile_size_seq))

        compute_D_matrix[grid](
            O, dO, D,
            O.stride(0), O.stride(1), O.stride(2),
            D.stride(0), D.stride(1),
            seq_len, d_v,
            tile_size_seq=ctx.tile_size_seq,
            tile_size_d=ctx.tile_size_d,
        )

        grid = (batch_dim, triton.cdiv(seq_len, ctx.tile_size_q))
        flash_attn_backward_triton[grid] (
            Q, K, V, O,
            dO,
            L, D,
            dQ, dK, dV,
            seq_len,
            Q.stride(0), Q.stride(1), Q.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            D.stride(0), D.stride(1), 1,
            tileQ=ctx.tile_size_q,
            tileKV=ctx.tile_size_kv,
            maxD=ctx.max_head_dim,
            is_causal=ctx.is_causal
        )



        # pytest.exit(0)
        return dQ, dK, dV, dCausal
