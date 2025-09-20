import argparse
import torch
import torch.cuda.nvtx as nvtx
import cs336_basics.model as model
from einops import rearrange, einsum
import timeit
from types import SimpleNamespace
import flash_attn
import torch._dynamo
import triton

def run_benchmark(q, k, v, fn, args):
    params = SimpleNamespace(**args)
    with nvtx.range("warmup"):
        with torch.autocast('cuda', dtype=params.dtype):
            for i in range(params.warmup):
                fn(q, k, v, mask=None)
    torch.cuda.synchronize()

    if params.memory_profile is not None:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    # forward bench
    start = timeit.default_timer()
    with nvtx.range("benchmark"):
        with torch.autocast('cuda', dtype=params.dtype):
            for i in range(params.num_reps):
                fn(q, k, v, mask=None)
    torch.cuda.synchronize()
    end = timeit.default_timer()

    elapsed = end - start
    fn_time = elapsed / params.num_reps
    return fn_time

class BenchmarkContext:
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype
        self.autocast_ctx = torch.autocast('cuda', dtype=self.dtype)

    def __enter__(self):
        self.autocast_ctx.__enter__()
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.autocast_ctx.__exit__(exc_type, exc_val, exc_tb)

def run_suite(params):
    batch_size = params.batch_size
    num_heads = params.num_heads
    context_length = params.context_length
    d_model = params.d_model


    print("##################################################################")
    print(f"########### Running suite [seq_len = {context_length} d_model = {d_model}] ###########")
    print("##################################################################")


    q = torch.randn(batch_size, num_heads, context_length, d_model, device='cuda', requires_grad=True)
    k = torch.randn(batch_size, num_heads, context_length, d_model, device='cuda', requires_grad=True)
    v = torch.randn(batch_size, num_heads, context_length, d_model, device='cuda', requires_grad=True)

    results = {
       "batch_size" : batch_size,
       "seq_len"  : seq_len,
       "d_model" : d_model,
       "dtype" : params.dtype
    }

    causal_mask = torch.tril(torch.ones(context_length, context_length)) == 1
    causal_mask = causal_mask.to('cuda')

    torch.set_float32_matmul_precision('high')
    torch._functorch.config.donated_buffer = False # for backward opt
    attn_compiled = torch.compile(model.scaled_dot_product_attention)

    with BenchmarkContext("forward naive", params.dtype) as name:
        if "forward_naive" in params.kernels:
            attention_fn = lambda:  model.scaled_dot_product_attention(q, k, v, mask=causal_mask)
            res = triton.testing.do_bench(attention_fn, warmup=params.warmup, rep=params.rep)
            print(f"Benchmark {name} took {res}")
            results[name] = res

    with BenchmarkContext("forward torch_compiled", params.dtype) as name:
        if "forward_torch_compiled" in params.kernels:
            attention_fn = lambda:  attn_compiled(q, k, v, mask=causal_mask)
            res = triton.testing.do_bench(attention_fn, warmup=params.warmup, rep=params.rep)
            print(f"Benchmark {name} took {res}")
            results[name] = res

    if len(q.shape) > 3:
        q = rearrange(q, "... seq d_k -> (...) seq d_k")
        k = rearrange(k, "... seq d_k -> (...) seq d_k")
        v = rearrange(v, "... seq d_v -> (...) seq d_v")

    with BenchmarkContext("forward flash_torch", params.dtype) as name:
        if "forward_flash_torch" in params.kernels:
            attention_fn = lambda: flash_attn.FlashAttentionTorch.apply(q, k, v, True)
            res = triton.testing.do_bench(attention_fn, warmup=params.warmup, rep=params.rep)
            print(f"Benchmark {name} took {res}")
            results[name] = res

    with BenchmarkContext("forward flash_triton", params.dtype) as name:
        if "forward_flash_triton" in params.kernels:
            attention_fn = lambda: flash_attn.FlashAttentionTriton.apply(q, k, v, True)
            res = triton.testing.do_bench(attention_fn, warmup=params.warmup, rep=params.rep)
            print(f"Benchmark {name} took {res}")
            results[name] = res

    with BenchmarkContext("backward naive", params.dtype) as name:
        if "backward_naive" in params.kernels:
            O = model.scaled_dot_product_attention(q, k, v, mask=causal_mask)
            loss = O.sum()
            attention_fn = lambda:  loss.backward(retain_graph=True)
            res = triton.testing.do_bench(attention_fn, warmup=params.warmup, rep=params.rep)
            print(f"Benchmark {name} took {res}")
            results[name] = res

    with BenchmarkContext("backward compiled", params.dtype) as name:
        if "backward_compiled" in params.kernels:
            O = attn_compiled(q, k, v, mask=causal_mask)
            loss = O.sum()
            attention_fn = lambda:  loss.backward(retain_graph=True)
            res = triton.testing.do_bench(attention_fn, warmup=params.warmup, rep=params.rep)
            print(f"Benchmark {name} took {res}")
            results[name] = res

    with BenchmarkContext("backward flash_torch", params.dtype) as name:
        if "backward_flash_torch" in params.kernels:
            O = flash_attn.FlashAttentionTorch.apply(q, k, v, True)
            loss = O.sum()
            attention_fn = lambda:  loss.backward(retain_graph=True)
            res = triton.testing.do_bench(attention_fn, warmup=params.warmup, rep=params.rep)
            print(f"Benchmark {name} took {res}")
            results[name] = res

    with BenchmarkContext("backward flash_triton", params.dtype) as name:
        if "backward_flash_triton" in params.kernels:
            O = flash_attn.FlashAttentionTriton.apply(q, k, v, True)
            loss = O.sum()
            attention_fn = lambda:  loss.backward(retain_graph=True)
            res = triton.testing.do_bench(attention_fn, warmup=params.warmup, rep=params.rep)
            print(f"Benchmark {name} took {res}")
            results[name] = res

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark configuration")
    _ = parser.add_argument("--batch", type=int, default="1")
    _ = parser.add_argument("--seq_len", type=str, default="128,256")
    _ = parser.add_argument("--d_model", type=str, default='128')
    _ = parser.add_argument("--warmup", type=float, default="0.1")
    _ = parser.add_argument("--rep", type=float, default="0.1")
    _ = parser.add_argument("--dtype", type=str, default="float32")
    all_models = [ 'forward_naive', 'forward_torch_compiled', 'forward_flash_torch',
       'forward_flash_triton', 'backward_naive', 'backward_compiled',
       'backward_flash_torch', 'backward_flash_triton',
    ]
    all_models_str = ','.join(all_models)
    _ = parser.add_argument("--kernels", type=str, default=all_models_str)

    args = parser.parse_args()

    params = {}
    params['dtype'] = getattr(torch, args.dtype)
    params['warmup'] = args.warmup * 1000.
    params['rep'] = args.rep * 1000.
    params['num_heads'] = 1
    params['batch_size'] = args.batch
    params['kernels'] = args.kernels.split(",")

    seq_len = [int(x) for x in args.seq_len.split(",")]
    d_model = [int(x) for x in args.d_model.split(",")]

    all_results = []
    for seq_len in seq_len:
        for head_dim in d_model:
            params['context_length'] = seq_len
            params['d_model'] = head_dim
            par = SimpleNamespace(**params)
            results = run_suite(par)
            print(results)
            all_results.append(results)

    print("all results")
    for result in all_results:
        print(result, end=',\n')
