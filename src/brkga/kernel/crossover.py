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
            unsigned int* indices, float* weight, float* random,
            float* population, float* output, unsigned int N,
            unsigned int pi_t) {
        // Find the thread indices
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int p = 0;

        for (int k = 0; k < N; k++) {
            float acc = 0;

            for (p = 0; p < pi_t; p++) {
                acc += weight[i*pi_t + p];
                if (acc <= random[i*N + k]) {
                    break;
                }
            }

            if (p == pi_t) p--;
            output[i*N + k] = population[indices[i*pi_t + p]*N + k];
        }
    }
''', 'crossover_mp')
