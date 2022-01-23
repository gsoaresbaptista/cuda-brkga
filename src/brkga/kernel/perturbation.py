import cupy as cp


perturbation = cp.RawKernel(r'''
    extern "C" __global__
    void perturbation(
            float* population, float* percentages, int* indices,
            unsigned int gene_size, float probability) {
        // Find the thread indices
        int i = blockDim.x * blockIdx.x + threadIdx.x;

        for (int k = 0; k < gene_size; k++) {
            if (percentages[i*gene_size + k] <= probability) {
                float tmp = population[i*gene_size + k];
                int idx = indices[i*gene_size + k];
                population[i*gene_size + k] = population[i*gene_size + idx];
                population[i*gene_size + idx] = tmp;
            }
        }
    }
''', 'perturbation')
