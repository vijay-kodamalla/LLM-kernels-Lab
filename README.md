# High-Performance CUDA Kernel Optimization Lab

A systematic exploration of hardware-aware kernel engineering for LLM/VLM architectures. 
This repository documents the optimization journey from baseline implementations to 
SOTA techniques on NVIDIA Turing (T4), Ampere (A100), and Hopper (H100) architectures.

## 🚀 Performance Summary

| Operator | Precision | Device | Baseline | Optimized | Speedup | Metric |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RMSNorm** | FP32 | Tesla T4 | 71.4 GB/s | 254.1 GB/s | 3.5x | Memory BW |
| **GEMM** | FP16/32| A100 | [TBD] | [TBD] | [TBD] | TFLOPS |
| **Softmax** | FP32 | T4 | [TBD] | [TBD] | [TBD] | Memory BW |

---

## 🛠 Project Workflow: Profile-Driven Optimization

For every kernel in this lab, I follow a strict engineering pipeline:
1. **Verification**: Compare GPU output against a CPU reference or cuBLAS to ensure `Status: PASS`.
2. **Profiling**: Use `ncu` (Nsight Compute) to identify hardware bottlenecks (Memory-bound vs. Compute-bound).
3. **Optimization**: Apply hardware-specific techniques (Vectorization, Tiling, Double Buffering, Tensor Cores).
4. **Validation**: Document the improvement in hardware utilization (e.g., SOL Throughput, Warp Occupancy).

## 📂 Kernel Index

### 1. [RMSNorm Optimization](./kernels/rmsnorm)
- **Baseline**: Initial fused implementation using warp-reductions.
- **Optimization**: Implemented 128-bit vectorized loads (`float4`) to maximize DRAM throughput.
- **Result**: Achieved 82.4% of peak DRAM bandwidth on Tesla T4.

### 2. [GEMM (General Matrix Multiplication)](./kernels/gemm)
- **S1**: Shared Memory Tiling.
- **S2**: Register Tiling & Outer-Product optimization.
- **S4**: Tensor Core acceleration via WMMA and PTX.

---
## 💻 Environment
- **Development**: WSL2 (Ubuntu 22.04), CUDA 12.x
- **Infrastructure**: Google Colab Pro (Tesla T4, A100, H100)
- **Profiling Tools**: NVIDIA Nsight Compute (ncu), Nsight Systems (nsys)
