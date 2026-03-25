# requirements-lock

Export lock snapshots after environment creation:

```bash
conda activate reviewers-sim-cpu
pip freeze > requirements-lock/cpu-freeze.txt

conda activate reviewers-sim-gpu
pip freeze > requirements-lock/gpu-freeze.txt
```

Notes:

- On Windows, `vllm` build and `faiss-gpu-cu12` wheels may be unavailable.
- For GPU inference with `vllm`/FAISS-GPU, use Linux + CUDA 12.1.
- CPU and GPU lock files in this folder reflect the current machine state.
- Linux-only optional GPU extras are documented in `envs/gpu-linux-extras.md`.
