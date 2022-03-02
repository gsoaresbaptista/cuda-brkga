import cupy as cp


local_search = cp.RawKernel(r'''
    __device__ inline float squared(float a) {
        return a * a;
    }

    __device__ inline float distance(float x1, float y1, float x2, float y2) {
        return sqrt(squared(x1 - x2) + squared(y1 - y2));
    }

    __device__ inline int get_route_id(int pos, unsigned int* population,
        float* info, float max_capacity, unsigned int gene_size) {
        //
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        float capacity = 0;

        int id;
        int actual_pos = 0;

        for (int j = 0; j < gene_size; j++) {
            id = population[gene_size*i + j];

            if (capacity + info[id*3 + 2] < max_capacity) {
                capacity += info[id*3 + 2];
            } else {
                if (pos == actual_pos)
                    return j;
                else
                    actual_pos++;
                capacity = info[id*3 + 2];
            }
        }
        return -1;
    }

    __device__ inline float evaluate(unsigned int* population,
        float* info, float max_capacity, unsigned int gene_size) {
        //
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        float fx = info[0], fy = info[1];
        float lx = info[0], ly = info[1];
        float output = 0, capacity = 0;

        int id;
        float x, y;

        for (int j = 0; j < gene_size; j++) {
            id = population[gene_size*i + j];
            x = info[id*3], y = info[id*3 + 1];

            if (capacity + info[id*3 + 2] < max_capacity) {
                output += distance(x, y, lx, ly);
                capacity += info[id*3 + 2];
            } else {
                output += distance(lx, ly, fx, fy);
                output += distance(x, y, fx, fy);
                capacity = info[id*3 + 2];
            }
            lx = x, ly = y;
        }

        output += distance(lx, ly, fx, fy);
        return output;
    }

    __device__ inline float copy_solution(
            unsigned int* population, unsigned int* output,
            unsigned int gene_size) {
        //
        int i = blockDim.x * blockIdx.x + threadIdx.x;

        for (int j = 0; j < gene_size; j++) {
            output[gene_size*i + j] = population[gene_size*i + j];
        }
    }

    extern "C" __global__
    void local_search(
            unsigned int* population, float* info, unsigned int* output,
            float max_capacity, unsigned int gene_size) {
        int cid = blockDim.x * blockIdx.x + threadIdx.x;

        // Evaluate the current solution
        float best = evaluate(population, info, max_capacity, gene_size);
        copy_solution(population, output, gene_size);

        // Split routes
        int id = 0, last_id = 0;

        for (int i = 0; id != -1; i++, last_id = id) {
            id = get_route_id(i, population, info, max_capacity, gene_size);

            // Optimize route
            for (int j = last_id; j < id; j++) {
                int tmp = population[gene_size*cid + j];
                float val;

                for (int k = j; k < id; k++) {
                    population[gene_size*cid+j] = population[gene_size*cid+k];
                    population[gene_size*cid+k] = tmp;
                    val = evaluate(population, info, max_capacity, gene_size);

                    if (val < best) {
                        copy_solution(population, output, gene_size);
                        best = val;
                    }

                    population[gene_size*cid+k] = population[gene_size*cid+j];
                    population[gene_size*cid+j] = tmp;
                }
            }
        }
    }
''', 'local_search')
