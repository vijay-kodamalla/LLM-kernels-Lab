#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

__device__ __forceinline__ float warpReduceSum(float val){
    for(int offset=16; offset>0; offset /= 2){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val){

    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    // this will return warp-level reduction sum, so now we have all values reduction values at warp-level
    val = warpReduceSum(val); 
    
    // only if the root thread is there, then you load the value to SMEM
    if(lane == 0) shared[warpId] = val;
    __syncthreads(); // to prevent any race conditions

    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if(warpId == 0) val = warpReduceSum(val);

    return val;
}

__global__ void rmsNorm_kernel(const float* input, float* output, const float* weight, int N, float eps){

    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    // grid-stride loop
    for(int i = tid; i < N; i += blockDim.x){
        float val = input[row * N + i];
        sum += val * val;
    }

    // the final block-level sum
    float final_sum = blockReduceSum(sum);

    // now we need to do a broadcast operation, so that all threads see the final block-level sum 
    __shared__ float inv_rms;

    if(tid == 0){
        inv_rms = rsqrtf(final_sum / N + eps);
    }

    __syncthreads();

    // now we kept the block-level sum in SMEM, now we can complete the final result
    for(int i = tid; i < N; i += blockDim.x){
        output[row * N + i] = (input[row * N + i] * inv_rms * weight[i]);
    }
}

// Harness to verify and time
int main() {
    const int N = 128, D = 4096;
    const float eps = 1e-5f;
    std::vector<float> h_in(N*D), h_weight(D), h_out(N*D);
    for(int i=0; i<N*D; i++) h_in[i] = (float)rand()/RAND_MAX;
    for(int i=0; i<D; i++) h_weight[i] = (float)rand()/RAND_MAX;

    float *d_in, *d_weight, *d_out;
    cudaMalloc(&d_in, N*D*sizeof(float));
    cudaMalloc(&d_weight, D*sizeof(float));
    cudaMalloc(&d_out, N*D*sizeof(float));

    cudaMemcpy(d_in, h_in.data(), N*D*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), D*sizeof(float), cudaMemcpyHostToDevice);

    rmsNorm_kernel<<<N, 256>>>(d_in, d_out, d_weight, D, eps);
    cudaDeviceSynchronize();
    
    printf("V1 Baseline: Kernel Executed.\n");
    cudaFree(d_in); cudaFree(d_weight); cudaFree(d_out);
    return 0;
}
