import cupy as cp
from brkga.problem import Problem
from .kernel import fitness as fitness_function


class CVRPProblem(Problem):
    def __init__(self, max_capacity: float) -> None:
        super().__init__()
        self.__max_capacity = max_capacity

    def decoder(
            self,
            population: cp.ndarray,
            population_size: int,
            gene_size: int) -> cp.ndarray:
        #
        return cp.argsort(population).astype(cp.uint32) + 1

    def fitness(
            self,
            population: cp.ndarray,
            info: cp.ndarray,
            population_size: int,
            gene_size: int) -> cp.ndarray:
        #
        values = cp.zeros((population_size,), dtype=cp.float32)

        #
        fitness_function(
            (population_size,), (1,),
            (population,
             info,
             values,
             cp.float32(self.__max_capacity),
             cp.uint32(gene_size)))

        return values
