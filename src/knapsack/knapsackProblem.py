import cupy as cp
from brkga.problem import Problem
from .kernel import decoder as decoder_function


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
        output = cp.zeros((population_size, gene_size), dtype=cp.float32)

        # decoder_function(
        #     self.bpg, self.tpb,
        #     (population, output, gene_size))

        # return output
        return cp.floor(population + 0.5)

    def fitness(
            self,
            population: cp.ndarray,
            info: cp.ndarray,
            population_size: int,
            gene_size: int) -> cp.ndarray:
        value = population.dot(info[:, 1])
        penalty = cp.power(cp.maximum(0, population.dot(info[:, 0]) - self.__max_weight), 2) * value
        return value - penalty
