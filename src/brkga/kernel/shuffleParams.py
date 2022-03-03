import cupy as cp


shuffle_params = cp.RawKernel(r'''
    extern "C" __global__
    void shuffle_params(
            unsigned int* indices, unsigned int* indices_out,
            float* weight, float* weight_out,
            unsigned int* id, unsigned int N) {
        // Find the thread indices
        int i = blockDim.x * blockIdx.x + threadIdx.x;

        for (int j = 0; j < N; j++) {
            indices_out[i*N + j] = indices[i*N + id[i*N + j]];
            weight_out[i*N + j] = weight[i*N + id[i*N + j]];
        }
    }
''', 'shuffle_params')
