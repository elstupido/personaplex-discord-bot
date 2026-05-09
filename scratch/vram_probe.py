import pynvml
import torch

try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"--- NVML REPORT ---")
    print(f"Total: {info.total / 1024**3:.2f} GB")
    print(f"Free:  {info.free / 1024**3:.2f} GB")
    print(f"Used:  {info.used / 1024**3:.2f} GB")
except Exception as e:
    print(f"NVML Error: {e}")

if torch.cuda.is_available():
    print(f"\n--- PYTORCH REPORT ---")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Memory Reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    free, total = torch.cuda.mem_get_info()
    print(f"Cuda Mem Get Info Free: {free / 1024**3:.2f} GB")
