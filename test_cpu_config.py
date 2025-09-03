#!/usr/bin/env python3
"""
Test script to verify CPU configuration works correctly
"""

import os
import torch

# Test the CPU detection logic
print("=== CPU Configuration Test ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"USE_CPU env var: {os.environ.get('USE_CPU', 'not set')}")

# Replicate the detection logic from rp_handler.py
USE_CPU = not torch.cuda.is_available() or os.environ.get('USE_CPU', 'false').lower() in ('true', '1', 'yes')
print(f"Will use CPU: {USE_CPU}")

if USE_CPU:
    print("✅ CPU mode will be used")
    print("   - TIREX_NO_CUDA=1 will be set")
    print("   - device_map='cpu'")
    print("   - torch_dtype=torch.float32")
else:
    print("✅ CUDA mode will be used")
    print("   - device_map='cuda'")
    print("   - torch_dtype=torch.bfloat16")

print("\n=== Environment Variables ===")
if USE_CPU:
    print("export TIREX_NO_CUDA=1")
    print("export USE_CPU=1  # optional override")
else:
    print("CUDA is available - no special environment variables needed")