# Lab 1: Fused RMSNorm Optimization

## 1. Problem Statement
RMSNorm (Root Mean Square Layer Normalization) is a critical operator in modern LLMs (e.g., Llama 3). It is **memory-bandwidth bound**, meaning the performance is limited by how fast we can move data from HBM to the ALUs. 

The goal of this lab was to:
1. Implement a **Fused** kernel to reduce HBM round-trips.
2. Optimize memory access patterns to reach the hardware's "Roofline" limit.

## 2. Implementations

### v1: Baseline Fused Kernel
- **Approach**: Uses a single kernel launch to perform square-sum, reduction, and normalization.
- **Communication**: Utilizes `__shfl_down_sync` for warp-level reductions and Shared Memory for block-level broadcasting.
- **Memory Access**: Standard 32-bit (scalar) loads.

### v2: Vectorized Kernel (Optimization)
- **Approach**: Implements **128-bit vectorized loads** using `float4`.
- **Reasoning**: Reduces the number of instructions handled by the Warp Scheduler, allowing for better saturation of the memory controllers.

## 3. Profiling & Performance Analysis
The following metrics were captured using **NVIDIA Nsight Compute (ncu)** on a **Tesla T4 GPU**.

| Metric | v1 (Baseline) | v2 (Vectorized) | Improvement |
| :--- | :--- | :--- | :--- |
| **DRAM Throughput** | 65.44% | **82.40%** | +16.96% |
| **Memory Bandwidth** | 201.53 GB/s | **254.07 GB/s** | +52.54 GB/s |
| **32-bit Mem Instructions** | 271,360 | **~0** | -100% |
| **128-bit Mem Instructions** | 0 | **67,840** | New |

### Key Insight:
The profiling of **v1** revealed a high count of `sm__sass_inst_executed_op_memory_32b`, indicating high instruction issue overhead. By moving to `float4` in **v2**, we reduced the instruction count by 4x, effectively shifting the bottleneck from the **Instruction Scheduler** to the **DRAM Bandwidth** limit.

## 4. Validation
- **Hardware**: NVIDIA Tesla T4 (Compute Capability 7.5)
- **Status**: `PASS`
- **Numerical Precision**: Max Absolute Error vs. CPU Reference: `2.38e-07`

## 5. How to Run
```bash
nvcc -O3 -arch=sm_75 rmsnorm_v2.cu -o rmsnorm_v2
./rmsnorm_v2
