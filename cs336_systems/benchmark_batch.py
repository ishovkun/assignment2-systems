import argparse
from ast import arg
import torch
from torch import Tensor
from jaxtyping import Float, Bool, Int
import math
import os

from einops import rearrange, einsum
import torch.cuda.nvtx as nvtx
import cs336_basics
from cs336_basics.nn_utils import softmax
import timeit
from cs336_basics.model import BasicsTransformerLM, Linear, silu
from cs336_basics.nn_utils import cross_entropy

@nvtx.range("attention")
def scaled_dot_product_attention_annotated(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    with nvtx.range("attn scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("attn softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    ret = None
    with nvtx.range("attn final matmul"):
        ret = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return ret

class SwiGLU_annotated(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    @nvtx.range("swiglu")
    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))

cs336_basics.model.scaled_dot_product_attention = scaled_dot_product_attention_annotated
cs336_basics.model.SwiGLU = SwiGLU_annotated


def create_model(size: str, context_len: int) -> BasicsTransformerLM:
    if size == "small":
        d_model = 768
        d_ff = 3072
        num_layers = 12
        num_heads = 12
    elif size == "medium":
        d_model = 1024
        d_ff = 4096
        num_layers = 24
        num_heads = 16
    elif size == "large":
        d_model = 1280
        d_ff = 5120
        num_layers = 36
        num_heads = 20
    elif size == "xl":
        d_model = 1600
        d_ff = 6400
        num_layers = 48
        num_heads = 25
    elif size == "2.7B":
        d_model = 2560
        d_ff = 10240
        num_layers = 32
        num_heads = 32
    else: raise ValueError(f"Invalid model size {size}")

    vocab_size=10_000
    rope_theta=10000.
    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta)

def get_random_batch(
    vocab_size: int,
    batch_size: int,
    context_length: int
) -> Int[Tensor, "batch seq_len"]:
    return torch.randint(low=0, high=vocab_size, size=(batch_size, context_length))


def run_benchmark(
    model: BasicsTransformerLM,
    batch_size,
    num_reps: int,
    num_warmup_reps: int = 0,
    memory_profile: str | None = None,
    dtype: torch.dtype = torch.float32
):
    x = get_random_batch(model.vocab_size, batch_size, model.context_length).cuda()
    y = get_random_batch(model.vocab_size, batch_size, model.context_length).cuda()

    with nvtx.range("warmup"):
        with torch.autocast('cuda', dtype=dtype):
            for i in range(num_warmup_reps):
                logits = model.forward(x)
                logits = logits.view(-1, logits.shape[-1])
                y = y.view(-1)

                loss = cross_entropy(logits, y)
                loss.backward()
    torch.cuda.synchronize()

    if memory_profile is not None:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    # forward bench
    start = timeit.default_timer()
    with nvtx.range("forward"):
        with torch.autocast('cuda', dtype=dtype):
            for i in range(num_reps):
                logits = model.forward(x)
    torch.cuda.synchronize()
    end = timeit.default_timer()

    elapsed = end - start
    forward_avg = elapsed / num_reps

    logits = model.forward(x)
    logits = logits.view(-1, logits.shape[-1])
    y = y.view(-1)

    loss = cross_entropy(logits, y)
    torch.cuda.synchronize()

    start = timeit.default_timer()
    with nvtx.range("backward"):
        with torch.autocast('cuda', dtype=dtype):
            for i in range(num_reps):
                loss.backward(retain_graph=True)
    torch.cuda.synchronize()
    end = timeit.default_timer()
    elapsed = end - start
    backward_avg = elapsed / num_reps

    if memory_profile is not None:
        torch.cuda.memory._dump_snapshot(memory_profile)
        torch.cuda.memory._record_memory_history(enabled=None)

    return forward_avg, backward_avg, forward_avg + backward_avg

def append_suffix(path: str, suffix: str) -> str:
    """
    Takes a file path and a suffix, strips the file extension,
    appends the suffix, and then adds back the file extension.
    """
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark configuration")
    parser.add_argument("model_size", type=str, default="small", nargs="?", help="model size = [small, medium, large, xl, 2.7B")
    parser.add_argument("--batch", type=int, default="4", help="Batch size")
    parser.add_argument("--context_length", type=int, default="256", help="Batch size")
    parser.add_argument("--warmup", type=int, default="5", help="Number of warmup steps")
    parser.add_argument("--reps", type=int, default="10", help="Number of bench steps")
    parser.add_argument("--mem", type=str, default=None, help="Memory profiler file")
    parser.add_argument("--dtype", type=str, default="float32", help="Auto quantization")
    print(type(torch.float32))


    args = parser.parse_args()
    dtype: torch.dtype = getattr(torch, args.dtype)

    sizes = args.model_size.split(",")
    if sizes[0] == "all": sizes = ["small", "medium", "large", "xl", "2.7B"]
    print(f"| Model | dtype | Num parameters [bil] | Time Forward [ms] | Time backward [ms] | Time Total [ms] |")
    print(f"|------- | ---- | --------------------- | ----------- | ----------- | ----------- |")
    for model_size in sizes:
        model_size = model_size.lstrip().rstrip()
        model = create_model(model_size, args.context_length)
        mem_profiler_output = args.mem
        if mem_profiler_output is not None:
            mem_profiler_output = append_suffix(mem_profiler_output, f"_{model_size}")

        tf, tb, tt = run_benchmark(model.cuda(),
            args.batch,
            num_reps=args.reps,
            num_warmup_reps=args.warmup,
            memory_profile=mem_profiler_output,
            dtype=dtype)
        print(f"|{model_size} | {dtype} | {(model.get_num_params()/1e9):2f} | {(tf*1000):.2f} | {(tb*1000):.2f} | {(tt*1000):.2f} |")
