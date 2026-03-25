# Linux-only Optional GPU Extras

Install these only on Linux CUDA 12.1 hosts:

```bash
conda activate reviewers-sim-gpu
pip install faiss-gpu-cu12
pip install vllm
```

Windows note:
- `faiss-gpu-cu12` and `vllm` wheels are typically unavailable on Windows.
