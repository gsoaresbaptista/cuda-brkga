import cupy as cp


crossover = cp.RawKernel(r'''
    extern "C" __global__
    void crossover(
            float* percentages, float* commons, float* elites,
            float* output, unsigned int N, unsigned int M, float pe) {
        // Find the thread indices
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        if (percentages[j + i*N] < pe) {
            output[j + i*N] = elites[j + i*N];
        } else {
            output[j + i*N] = commons[j + i*N];
        }
    }
''', 'crossover')

crossover_mp = cp.RawKernel(r'''
    extern "C" __global__
    void crossover_mp(
            float* percentages, float* commons, float* elites,
            float* output, unsigned int N, unsigned int M, float pe) {
        // Find the thread indices
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        if (percentages[j + i*N] < pe) {
            output[j + i*N] = elites[j + i*N];
        } else {
            if (percentages[j + i*N] < (1-pe)/2.0) {
                output[j + i*N] = commons[j + i*2*N];
            } else {
                output[j + i*N] = commons[j + i*2*N + N];
            }
        }
    }
''', 'crossover_mp')
