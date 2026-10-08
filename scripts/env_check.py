import subprocess
import sys

print("=== versions ===")
for m in ["torch", "transformers", "trl", "peft", "bitsandbytes", "datasets", "unsloth", "accelerate", "xformers"]:
    try:
        mod = __import__(m)
        print(f"  {m:16s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:16s} IMPORT FAIL {type(e).__name__}: {e}")

import torch

print("\n=== gpu ===")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} {p.name} {p.total_memory/1e9:.1f}GB cc={p.major}.{p.minor}")

print("\n=== bf16 truth on T4 ===")
print("  is_bf16_supported() [permissive] :", torch.cuda.is_bf16_supported())
try:
    print("  including_emulation=False       :", torch.cuda.is_bf16_supported(including_emulation=False))
except TypeError as e:
    print("  including_emulation kwarg unsupported:", e)
