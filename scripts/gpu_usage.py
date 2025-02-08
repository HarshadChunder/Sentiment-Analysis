import torch
import psutil
import os
import gc

def check_cuda():
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))

def print_system_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[System Memory] RSS (Resident Set Size): {mem_info.rss / (1024 ** 2):.2f} MB")
    print(f"[System Memory] VMS (Virtual Memory Size): {mem_info.vms / (1024 ** 2):.2f} MB")

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[GPU Memory] Allocated: {allocated:.2f} MB")
        print(f"[GPU Memory] Reserved: {reserved:.2f} MB")

def list_tensors():
    if torch.cuda.is_available():
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor):
                print(f"Tensor - Size: {obj.size()}, Memory: {obj.element_size() * obj.nelement() / (1024 ** 2):.2f} MB, Device: {obj.device}")