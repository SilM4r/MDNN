#include <cuda_runtime.h>


extern "C" __declspec(dllexport) void LayerCalculation(float* a, float* b, float* c, float* result, int size, int quantity);

// nvcc -shared -o gpu.dll GPU.cu
// Kernel function for vector addition
__global__ void LayerCalculationKernel(float* values, float* weights,float* bias , float* result, int size, int quantity) {
    float sum = 0.0f;  
    int i = threadIdx.x + blockDim.x * blockIdx.x; 

    if(i < quantity) 
    {  
        for (int k = 0; k < size; k++) {
            sum += values[k] * weights[i * size + k];   
        }

        result[i] = sum + bias[i]; 
    }
}


// Exported function called from C#
extern "C" __declspec(dllexport) void LayerCalculation(float* values, float* weights, float* bias, float* result, int size, int quantity) {
    float* d_values;
    float* d_weights;
    float* d_bias;
    float* d_result;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_values, size * sizeof(float));
    cudaMalloc((void**)&d_bias, quantity * sizeof(float));
    cudaMalloc((void**)&d_weights, size * quantity * sizeof(float));
    cudaMalloc((void**)&d_result, quantity * sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(d_values, values, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, quantity * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, size * quantity * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel with 256 threads per block
    int blockSize = 256;
    int numBlocks = (quantity + blockSize - 1) / blockSize;
    LayerCalculationKernel<<<numBlocks, blockSize>>>(d_values, d_weights,d_bias, d_result, size, quantity);

    // Copy the results back to the CPU
    cudaMemcpy(result, d_result, quantity * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_values);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_result);
}