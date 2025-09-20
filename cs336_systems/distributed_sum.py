import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import sys

def sizeof(dtype):
    return torch.tensor([0], dtype=dtype).element_size()

def setup(rank, world_size):
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "29501"
   backend = "gloo"
   dist.init_process_group(backend, rank=rank, world_size=world_size)
   # avoid specifying which out of 8 cuda gpus to actually use
   if backend == "nccl": torch.cuda.set_device(rank)

def distributed_demo(rank, world_size, tensor_size_mb):
    setup(rank, world_size)

    mb = 1024 * 1024
    num_elements = (tensor_size_mb * mb) // sizeof(torch.float32)
    data = torch.randint(0, 10, (num_elements,))
    dist.all_reduce(data, async_op=False)

def benchmark_reduce():
    setup(rank, world_size)
    mb = 1024 * 1024
    num_elements = (tensor_size_mb * mb) // sizeof(torch.float32)
    data = torch.randint(0, 10, (num_elements,))
    dist.all_reduce(data, async_op=False)

def main():
    world_size = 4
    tensor_size_mb = 1
    mp.spawn(fn=distributed_demo, args=(world_size, tensor_size_mb), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
