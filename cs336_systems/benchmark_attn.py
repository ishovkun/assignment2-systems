import argparse
import torch
import torch.cuda.nvtx as nvtx
import cs336_basics.model as model
import timeit
from types import SimpleNamespace
import torch._dynamo

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark configuration")
    parser.add_argument("model_size", type=str, default="small", nargs="?", help="model size = [small, medium, large, xl, 2.7B")
    parser.add_argument("--batch", type=int, default="4", help="Batch size")
    parser.add_argument("--context_length", type=int, default="256", help="Batch size")
    parser.add_argument("--warmup", type=int, default="5", help="Number of warmup steps")
    parser.add_argument("--reps", type=int, default="10", help="Number of bench steps")
    parser.add_argument("--mem", type=str, default=None, help="Memory profiler file")
    parser.add_argument("--dtype", type=str, default="float32", help="Auto quantization")

    args = parser.parse_args()

    params : dict = {}
    params['dtype'] = getattr(torch, args.dtype)
    params['warmup'] = args.warmup
    params['memory_profile'] = args.mem
    params['num_reps'] = args.reps


    sizes = args.model_size.split(",")
    if sizes[0] == "all": sizes = ["small", "medium", "large", "xl", "2.7B"]
    # print(f"| Model | dtype | Num parameters [bil] | Time Forward [ms] | Time backward [ms] | Time Total [ms] |")
    # print(f"|------- | ---- | --------------------- | ----------- | ----------- | ----------- |")
    for model_size in sizes:
        model_size = model_size.lstrip().rstrip()
        print(f"##########\nRunning model {model_size}\n##########")

        batch_size = args.batch
        context_length = args.context_length
        if model_size == "small":
            d_model = 768
            num_heads = 12
        elif model_size == "medium":
            d_model = 1024
            num_heads = 16
        elif model_size == "large":
            d_model = 1280
            num_heads = 20
        elif model_size == "xl":
            num_heads = 25
            d_model = 1600
        elif model_size == "2.7B":
            num_heads = 32
            d_model = 2560
        else:
            raise NotImplementedError


        q = torch.randn(batch_size, num_heads, context_length, d_model, device='cuda', requires_grad=True)
        k = torch.randn(batch_size, num_heads, context_length, d_model, device='cuda', requires_grad=True)
        v = torch.randn(batch_size, num_heads, context_length, d_model, device='cuda', requires_grad=True)


        report: dict[str, float] = {}
        report['naive'] = run_benchmark(q, k, v, model.scaled_dot_product_attention, params)
        print(f"Benchmark 'naive' took {report['naive']}")

        torch.set_float32_matmul_precision('high')
        attn_compiled = torch.compile(model.scaled_dot_product_attention)

        report['torch_compiled'] = run_benchmark(q, k, v, attn_compiled, params)
        print(f"Benchmark 'torch_compiled' took {report['torch_compiled']}")

        # print(report)

        speedup = report['naive'] / report['torch_compiled']
        print(f"speedup = {speedup}")
        # model = create_model(model_size, args.context_length)
        # mem_profiler_output = args.mem
        # # if mem_profiler_output is not None:
        #     mem_profiler_output = append_suffix(mem_profiler_output, f"_{model_size}")

        # print(f"|{model_size} | {dtype} | {(model.get_num_params()/1e9):2f} | {(tf*1000):.2f} | {(tb*1000):.2f} | {(tt*1000):.2f} |")
