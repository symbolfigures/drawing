#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

extern "C" void blendImageCUDA(unsigned char *input, unsigned char *output, int width, int height);

__global__ void blendKernel(unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int sumR = 0, sumG = 0, sumB = 0;
    int count = 0;

    // Iterate through the 3x3 grid
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;

            // Check bounds
            if (nx >= 0 && ny >= 0 && nx < width && ny < height) {
                int idx = (ny * width + nx) * 3;
                sumR += input[idx];
                sumG += input[idx + 1];
                sumB += input[idx + 2];
                ++count;
            }
        }
    }

    // Average the values
    int idx = (y * width + x) * 3;
    output[idx] = sumR / count;
    output[idx + 1] = sumG / count;
    output[idx + 2] = sumB / count;
}

extern "C" void blendImageCUDA(unsigned char *input, unsigned char *output, int width, int height) {
    unsigned char *d_input, *d_output;
    size_t size = width * height * 3 * sizeof(unsigned char);

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    blendKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
