from einops.einops import Tensor
import torch
from jaxtyping import Float, Bool, Int
import triton
import triton.language as tl
from einops import rearrange

def weighted_sum(x: Float[torch.Tensor, "... d"], weight: Float[torch.Tensor, "d"]):
    return (weight * x).sum(axis=-1)
    # return torch.sum(weight * x, axis=-1)

@triton.jit
def weighted_sum_fwd(
   x_ptr, weight_ptr, output_ptr,
   x_stride_row, x_stride_dim,
   weight_stride_dim,
   output_stride_row,
   ROWS, D,
   ROWS_TILE_SIZE: tl.constexpr,
   D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0) # blockIdx.x

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE,),
        order=(1, 0)
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # Initialize a buffer to write to
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # Load current block ptr
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # ROW_TILE_SIZE, D_TILE_SIZE
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE)

        # compute the weighted sum of the row
        output += tl.sum(row * weight[None, :], axis=1)

        # move the ptrs to the next tile
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE,))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    # Write output to the output block ptr (single scalar per row)
    tl.store(output_block_ptr, output, boundary_check=(0,))

@triton.jit
def weighted_sum_backward(
    x_ptr: Float[Tensor, 'R D'], weight_ptr: Float[Tensor, "D"], # ptrs to the original x and w data; x[R D], w[D]
    grad_output_ptr: Float[Tensor, 'R'], # gradient from downstream of the chain rule
    grad_x_ptr: Float[Tensor, 'R D'], partial_grad_weight_ptr : Float[Tensor, 'RT D'], # output ptrs
    stride_xr: int, stride_xd: int,
    stride_wd: int,
    stride_gr: int,
    stride_gxr: int, stride_gxd: int,
    stride_gwb: int, stride_gwd: int,
    NUM_ROWS: int, D: int,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0) # gridDim.x

    # Inputs
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,), strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D,), strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1,0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,), strides=(stride_wd,),
        offsets=(0,), block_shape=(D_TILE_SIZE),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D,), strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D,), strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0)
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero") # (ROWS_TILE_SIZE,)

        # outer product for grad_x
        weight = tl.laod(weight_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE,)
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boudnary_check=(0, 1))

        # Reduce as many rows as possible for the grad_weight result
        row = tl.load(x_block_ptr, boundary_check=(0, 1))
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,)) # neven out of boudns for dim 0

        # move pointers to the next tile along D
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # cache x and w to be used in the backward pass,
        # when we only receive the gradient wrt the output tensor, and
        # need to compute the gradients wrt x and wweight.
        D, output_dims = x.shape[-1], x.shape[:-1]

        # reshape input tensor to 2D
        input_shape = x.shape

        x = rearrange(x, "... d -> (...) d")

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D)
        ctx.ROWS_TILE_SIZE = 16

        # initialize empty result tensor. Note that these elements are not
        # necessarily 0
        y = torch.empty(output_dims, device=x.device)

        # launch our kernel with n instances in our 1D grid
        n_rows = y.numel()
        grid = (triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)
        weighted_sum_fwd[grid] (
            x, weight, y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.DILE_SIZE
        n_rows, D = x.shape

        # Each cta first writes to a partial buffer,
        # then reduces over this buffer to get the final grad
        buf_shape = (triton.cdiv(n_rows, ROWS_TILE_SIZE),)
        partial_grad_weight = torch.empty(buf_shape, device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        grid = (triton.cdiv(n_rows, ROWS_TILE_SIZE),)
        weighted_sum_backward[grid] (
            x, weight, grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight

if __name__ == "__main__":
    D = 6000
    B = 600
    x = torch.randn(B, D, device='cuda', requires_grad=True)
    w = torch.randn(D, device='cuda', requires_grad=True)
    print(f"x.stride(0) = {x.stride(0)}")
    print(f"x.stride(1) = {x.stride(1)}")


    # z = WeightedSumFunc.apply(x, w)
    # print(z[:5])
    # y = weighted_sum(x, w)
    # print(y[:5])
    # z.backward()
