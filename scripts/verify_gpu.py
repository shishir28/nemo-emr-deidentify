
import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}           : {props.name}")
        print(f"  Memory        : {props.total_memory / 1024**3:.1f} GB")
        print(f"  CUDA caps     : {props.major}.{props.minor}")
else:
    print("No GPU detected — check CUDA installation")
