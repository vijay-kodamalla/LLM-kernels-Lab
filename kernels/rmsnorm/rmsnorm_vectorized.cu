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

    val = warpReduceSum(val); 
    if(lane == 0) shared[warpId] = val;
    __syncthreads(); 

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if(warpId == 0) val = warpReduceSum(val);

    return val;
}

__global__ void rmsNorm_kernel_v2(const float* input, float* output, const float* weight, int N, float eps){

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Cast to float4 for 128-bit efficiency
    const float4* input4 = (const float4*)(input + row * N);
    float4* output4 = (float4*)(output + row * N);
    const float4* weight4 = (const float4*)weight;

    int N4 = N / 4;

    float sum = 0.0f;
    // grid-stride loop
    for(int i = tid; i < N4; i += blockDim.x){
        float4 val4 = input4[i];
        
        // now math on all 4 float vals
        sum += val4.x * val4.x;
        sum += val4.y * val4.y;
        sum += val4.z * val4.z;
        sum += val4.w * val4.w;
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
    for(int i = tid; i < N4; i += blockDim.x){
        float4 in_vals = input4[i];
        float4 w_vals = weight4[i];
        float4 result;

        result.x = (in_vals.x * inv_rms) * w_vals.x;
        result.y = (in_vals.y * inv_rms) * w_vals.y;
        result.z = (in_vals.z * inv_rms) * w_vals.z;
        result.w = (in_vals.w * inv_rms) * w_vals.w;

        output4[i] = result;
    }
}

int main() {
    const int N = 128, D = 4096;
    const float eps = 1e-5f;
    std::vector<float> h_in(N*D), h_weight(D);
    for(int i=0; i<N*D; i++) h_in[i] = (float)rand()/RAND_MAX;
    for(int i=0; i<D; i++) h_weight[i] = (float)rand()/RAND_MAX;

    float *d_in, *d_weight, *d_out;
    cudaMalloc(&d_in, N*D*sizeof(float));
    cudaMalloc(&d_weight, D*sizeof(float));
    cudaMalloc(&d_out, N*D*sizeof(float));

    cudaMemcpy(d_in, h_in.data(), N*D*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), D*sizeof(float), cudaMemcpyHostToDevice);

    rmsNorm_kernel_v2<<<N, 256>>>(d_in, d_out, d_weight, D, eps);
    cudaDeviceSynchronize();
    
    printf("V2 Vectorized: Kernel Executed.\n");
    cudaFree(d_in); cudaFree(d_weight); cudaFree(d_out);
    return 0;
}
