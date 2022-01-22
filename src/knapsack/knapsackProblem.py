import cupy as cp
from brkga.problem import Problem


class KnapsackProblem(Problem):
    def __init__(self, max_weight: int) -> None:
        self.__max_weight = max_weight
        super().__init__()

    def decoder(
            self,
            population: cp.ndarray,
            population_size: int,
            gene_size: int) -> cp.ndarray:
        #
        return cp.floor(population + 0.5)

    def fitness(
            self,
            population: cp.ndarray,
            info: cp.ndarray,
            population_size: int,
            gene_size: int) -> cp.ndarray:
        #
        value = population.dot(info[:, 1])
        penalty = cp.maximum(0, (population.dot(info[:, 0])
                             - self.__max_weight)) * value

        return value - penalty
