#include <cuda_runtime.h>
#include <iostream>
#include <complex.h>
#include <math.h>
#include <string.h>

#include <cuComplex.h>

using namespace std;
#define CHECK(condition, err_fmt, err_msg) \
    if (condition) { \
        printf(err_fmt " (%s:%d)\n", err_msg, __FILE__, __LINE__); \
        return EXIT_FAILURE; \
    }

#define CHECK_CUDA(stat) \
    CHECK((stat) != cudaSuccess, "CUDA error %s", cudaGetErrorString(stat))



__device__ uint32_t reverse_bits_gpu(uint32_t x)
{
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

__global__ void fft_kernel(const cuFloatComplex* x, cuFloatComplex* Y, uint32_t N, int logN)
{
    // Find this thread's index in the input array.
    uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    // Start by bit-reversing the input.
    // Reverse the 32-bit index.
    // Only keep the last logN bits of the output.
    uint32_t rev;

    rev = reverse_bits_gpu(2 * i);
    rev = rev >> (32 - logN);
    Y[2 * i] = x[rev];

    rev = reverse_bits_gpu(2 * i + 1);
    rev = rev >> (32 - logN);
    Y[2 * i + 1] = x[rev];

    __syncthreads();

    // Set mh to 1, 2, 4, 8, ..., N/2
    for (int s = 1; s <= logN; s++) {
        int mh = 1 << (s - 1);  // 2 ** (s - 1)

        // k = 2**s * (2*i // 2**(s-1))  for i=0..N/2-1
        // j = i % (2**(s - 1))  for i=0..N/2-1
        int k = threadIdx.x / mh * (1 << s);
        int j = threadIdx.x % mh;
        int kj = k + j;

        cuFloatComplex a = Y[kj];

        // exp(-2i pi j / 2**s)
        // exp(-2i pi j / m)
        // exp(-i pi j / (m/2))
        // exp(ix)
        // cos(x) + i sin(x)
        float tr;
        float ti;

        // TODO possible optimization:
        // pre-compute twiddle factor array
        // twiddle[s][j] = exp(-i pi * j / 2**(s-1))
        // for j=0..N/2-1 (proportional)
        // for s=1..log N
        // need N log N / 2 tmp storage...

        // Compute the sine and cosine to find this thread's twiddle factor.
        sincosf(-(float)M_PI * j / mh, &ti, &tr);
        cuFloatComplex twiddle = make_cuFloatComplex(tr, ti);

        cuFloatComplex b = cuCmulf(twiddle, Y[kj + mh]);

        // Set both halves of the Y array at the same time
        Y[kj] = cuCaddf(a, b);
        Y[kj + mh] = cuCsubf(a, b);

        // Wait for all threads to finish before traversing the array once more.
        __syncthreads();
    }
}

int fft_gpu(const cuFloatComplex* x, cuFloatComplex* Y, uint32_t N)
{
    // if N>0 is a power of 2 then
    // N & (N - 1) = ...01000... & ...00111... = 0
    // otherwise N & (N - 1) will have a 0 in it
    if (N & (N - 1)) {
        fprintf(stderr, "N=%u must be a power of 2.  "
                "This implementation of the Cooley-Tukey FFT algorithm "
                "does not support input that is not a power of 2.\n", N);

        return -1;
    }

    int logN = (int) log2f((float) N);

    cudaError_t st;

    // Allocate memory on the CUDA device.
    cuFloatComplex* x_dev;
    cuFloatComplex* Y_dev;
    st = cudaMalloc((void**)&Y_dev, sizeof(*Y) * N);
    // Check for any CUDA errors
    CHECK_CUDA(st);

    st = cudaMalloc((void**)&x_dev, sizeof(*x) * N);
    CHECK_CUDA(st);

    // Copy input array to the device.
    st = cudaMemcpy(x_dev, x, sizeof(*x) * N, cudaMemcpyHostToDevice);
    CHECK_CUDA(st);

    // Send as many threads as possible per block.
    int cuda_device_ix = 0;
    cudaDeviceProp prop;
    st = cudaGetDeviceProperties(&prop, cuda_device_ix);
    CHECK_CUDA(st);

    // Create one thread for every two elements in the array 
    int size = N >> 1;
    int block_size = min(size, prop.maxThreadsPerBlock);
    dim3 block(block_size, 1);
    dim3 grid((size + block_size - 1) / block_size, 1);

    // Call the kernel
    fft_kernel <<< grid, block >>> (x_dev, Y_dev, N, logN);

    // Copy the output
    st = cudaMemcpy(Y, Y_dev, sizeof(*x) * N, cudaMemcpyDeviceToHost);
    CHECK_CUDA(st);

    // Free CUDA memory
    st = cudaFree(x_dev);
    CHECK_CUDA(st);
    st = cudaFree(Y_dev);
    CHECK_CUDA(st);

    return EXIT_SUCCESS;
}


int main() {

    int runtimeVersion = 0;
    cudaRuntimeGetVersion(&runtimeVersion);
    cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "." << (runtimeVersion % 1000) / 10 << endl;
    // Define the size of the FFT, must be a power of 2
    const uint32_t N = 256;

    // Allocate and initialize the input array on the host
    cuFloatComplex x[N];
    for (uint32_t i = 0; i < N; ++i) {
        x[i] = make_cuFloatComplex(static_cast<float>(i), 0.0f); // real part is i, imaginary part is 0
    }

    for (uint32_t i = 0; i < N; ++i){
        printf("%f + %f i \n",cuCrealf(x[i]),cuCimagf(x[i]));
       
    }
    // Allocate output array
    cuFloatComplex Y[N];

    // Call the FFT function
    int result = fft_gpu(x, Y, N);

    if (result == EXIT_SUCCESS) {
        // Print the output
        std::cout << "FFT result:\n";
        for (uint32_t i = 0; i < N; ++i) {
            std::cout << "Y[" << i << "] = (" << cuCrealf(Y[i]) << ", " << cuCimagf(Y[i]) << ")\n";
        }
    } else {
        std::cerr << "FFT computation failed.\n";
    }

    return 0;
}
