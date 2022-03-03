import cupy as cp


crossover = cp.RawKernel(r'''
    extern "C" __global__
    void crossover(
            float* percentages, float* commons, int* commons_idx,
            float* elites, int* elites_idx,
            float* output, unsigned int N, float pe) {
        // Find the thread indices
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        if (percentages[j*N + i] < pe) {
            output[j*N + i] = elites[elites_idx[j]*N + i];
        } else {
            output[j*N + i] = commons[commons_idx[j]*N + i];
        }
    }
''', 'crossover')

crossover_mp = cp.RawKernel(r'''
    extern "C" __global__
    void crossover_mp(
            float* percentages, float* commons, int* commons_idx,
            float* elites, int* elites_idx,
            float* output, unsigned int N, float pe) {
        // Find the thread indices
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;

        if (percentages[j*N + i] < pe) {
            output[j*N + i] = elites[elites_idx[j]*N + i];
        } else {
            if (percentages[j*N + i] < (1-pe)/2.0) {
                output[j*N + i] = commons[commons_idx[j*2]*N + i];
            } else {
                output[j*N + i] = commons[commons_idx[j*2 + 1]*N + i];
            }
        }
    }
''', 'crossover_mp')
