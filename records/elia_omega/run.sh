#!/bin/bash

# MIT License
# Copyright (c) 2026 Igor Labadin
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# ==============================================================================
# Execution Script: Elia Omega v4.1 (The SOTA Killer)
# Target Hardware: 8xH100
# Target Dataset: Fixed FineWeb (Challenge constraints)
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

echo "🔥 [INIT] Bootstrapping Elia Omega v4.1 on 8xH100..."

# 1. Environment Variables for strict rule compliance
export DATA_PATH="./data/datasets/fineweb10B_sp8192"
export TOKENIZER_PATH="./data/tokenizers/fineweb_8192_bpe.model"
export MAX_WALLCLOCK_SECONDS=600.0 # Strict 10-minute compute cap enforcement

# Disable PyTorch multithreading overhead to maximize GPU saturation
export OMP_NUM_THREADS=4 

# 2. Launch Distributed Training (DDP)
# This will spawn 8 processes (one for each H100 GPU) and synchronize their gradients.
echo "🚀 [LAUNCH] Starting 8-GPU distributed training via torchrun. Time cap: 10 minutes."

torchrun --standalone --nproc_per_node=8 train.py

echo "✅ [DONE] Training session complete. Artifact 'final_model.int6.ptz' successfully generated."
