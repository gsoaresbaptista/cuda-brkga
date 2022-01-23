import cupy as cp


fitness = cp.RawKernel(r'''
    __device__ inline float squared(float a) {
        return a * a;
    }

    __device__ inline float distance(float x1, float y1, float x2, float y2) {
        return sqrt(squared(x1 - x2) + squared(y1 - y2));
    }

    extern "C" __global__
    void fitness(
            unsigned int* population, float* info, float* output,
            float max_capacity, unsigned int gene_size) {
        //
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        float fx = info[0], fy = info[1];
        float lx = info[0], ly = info[1];
        float total = 0, capacity = 0;

        int id;
        float x, y;

        for (int j = 0; j < gene_size; j++) {
            id = population[gene_size*i + j];
            x = info[id*3], y = info[id*3 + 1];

            if (capacity + info[id*3 + 2] < max_capacity) {
                total += distance(x, y, lx, ly);
                capacity += info[id*3 + 2];
            } else {
                total += distance(lx, ly, fx, fy);
                total += distance(x, y, fx, fy);
                capacity = info[id*3 + 2];
            }
            lx = x, ly = y;
        }

        total += distance(lx, ly, fx, fy);
        output[i] = total;
    }
''', 'fitness')
