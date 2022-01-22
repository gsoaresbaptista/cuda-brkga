import cupy as cp

decoder = cp.RawKernel(r'''
    extern "C" __global__
    void decoder(float* population, float* output, unsigned int N) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        if (population[j + i*N] >= 0.5)
            output[j + i*N] = 1;
        else
            output[j + i*N] = 0;
    }
''', 'decoder')
