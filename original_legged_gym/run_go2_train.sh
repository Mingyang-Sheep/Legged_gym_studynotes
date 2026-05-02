#!/bin/bash
# Run Go2 training with RTX 4090 nvrtc fix
export TORCH_CUDA_ARCH_LIST="8.6;8.9"
echo "TORCH_CUDA_ARCH_LIST = $TORCH_CUDA_ARCH_LIST"
python legged_gym/scripts/train.py --task=go2 "$@"
