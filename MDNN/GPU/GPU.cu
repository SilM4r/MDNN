#include <cuda_runtime.h>


extern "C" __declspec(dllexport) void LayerCalculation(float* a, float* b, float* c, float* result, int size, int quantity);

// nvcc -shared -o gpu.dll GPU.cu
// Kernel funkce pro sčítání vektorů
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


// Exportovaná funkce, která bude volána z C#
extern "C" __declspec(dllexport) void LayerCalculation(float* values, float* weights, float* bias, float* result, int size, int quantity) {
    float* d_values;
    float* d_weights;
    float* d_bias;
    float* d_result;

    // Alokace paměti na GPU
    cudaMalloc((void**)&d_values, size * sizeof(float));
    cudaMalloc((void**)&d_bias, quantity * sizeof(float));
    cudaMalloc((void**)&d_weights, size * quantity * sizeof(float));
    cudaMalloc((void**)&d_result, quantity * sizeof(float));

    // Kopírování dat z CPU do GPU
    cudaMemcpy(d_values, values, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, quantity * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, size * quantity * sizeof(float), cudaMemcpyHostToDevice);

    // Volání kernelu s 256 vlákny na blok
    int blockSize = 256;
    int numBlocks = (quantity + blockSize - 1) / blockSize;
    LayerCalculationKernel<<<numBlocks, blockSize>>>(d_values, d_weights,d_bias, d_result, size, quantity);

    // Kopírování výsledků zpět na CPU
    cudaMemcpy(result, d_result, quantity * sizeof(float), cudaMemcpyDeviceToHost);

    // Uvolnění paměti na GPU
    cudaFree(d_values);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_result);
}