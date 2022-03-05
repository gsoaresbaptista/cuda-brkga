import cupy as cp
from brkga.problem import Problem
from .kernel import fitness as fitness_function
from .kernel import local_search as local_search_function


class CVRPProblem(Problem):
    def __init__(self, max_capacity: float, local_search: int) -> None:
        super().__init__()
        self.__max_capacity = max_capacity
        self.__local_search = local_search

    def decoder(
            self,
            population: cp.ndarray,
            population_size: int,
            gene_size: int) -> cp.ndarray:
        #
        return cp.argsort(population).astype(cp.uint32) + 1

    def local_search(
            self,
            population: cp.ndarray,
            info: cp.ndarray,
            population_size: int,
            gene_size: int,
            generation: int) -> cp.ndarray:
        #
        if generation % self.__local_search == 0:
            values = cp.zeros(population.shape, dtype=cp.uint32)

            local_search_function(
                (population_size,), (1,),
                (population,
                 info,
                 values,
                 cp.float32(self.__max_capacity),
                 cp.uint32(gene_size)))

            return values
        return population

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
