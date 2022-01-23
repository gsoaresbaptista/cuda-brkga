import cupy as cp


crossover = cp.RawKernel(r'''
    extern "C" __global__
    void crossover(
            float* percentages, float* commons, float* elites,
            float* output, unsigned int N, unsigned int M, float pe) {
        // Find the thread indices
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        //if (i < M && j < N) {
            if (percentages[j + i*N] < pe) {
                output[j + i*N] = elites[j + i*N];
            } else {
                output[j + i*N] = commons[j + i*N];
            }
        //}
    }
''', 'crossover')
