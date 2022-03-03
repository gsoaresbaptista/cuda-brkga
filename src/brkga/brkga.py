import cupy as cp
from typing import List
from tqdm.autonotebook import tqdm
from .problem import Problem
from .kernel import crossover, crossover_mp
from colorama import Fore, Style


class BRKGA:
    def __init__(
            self,
            problem: Problem,
            gene_size: int,
            mp: bool = False,
            maximize: bool = True) -> None:
        self.__problem = problem
        self.__population_size = 0
        self.__elite_population = 0
        self.__mutants_population = 0
        self.__rest_population = 0
        self.__rhoe = 0.0
        self.__info = cp.zeros(0, dtype=cp.float32)
        self.__population = cp.zeros(0, dtype=cp.float32)
        self.__maximize = maximize
        self.__tpb = (0, 0)
        self.__bpg = (0, 0)
        self.__best_value = 0
        self.__best_individual = None
        self.__gene_size = gene_size
        self.__mp = mp

    @property
    def best_value(self) -> float:
        return self.__best_value

    @property
    def best_individual(self) -> cp.ndarray:
        return self.__best_individual

    def set_seed(self, seed: int) -> None:
        cp.random.seed(seed)

    def fit_population(
            self, p: int,
            pe: float,
            pm: float,
            rhoe: float) -> None:
        # Parameters
        self.__rhoe = rhoe
        self.__population_size = p
        self.__elite_population = int(p * pe)
        self.__mutants_population = int(p * pm)
        self.__rest_population = int(p * (1.0 - pe - pm))

    def fit_input(self, info: List) -> None:
        if self.__rhoe == 0.0:
            raise Exception(
                "Set population parameters before fitting to input.")

        self.__info = cp.array(info, dtype=cp.float32)
        self.__population = cp.random.uniform(
            low=0, high=1,
            size=(self.__population_size, self.__gene_size),
            dtype=cp.float32)

        # Cuda params
        tpb = (32, 32) if self.__info.shape[0] >= 32 else (1, 1)
        bpg = (self.__population_size // tpb[0] + 1,
               self.__gene_size // tpb[0] + 1)

        self.__problem.tpb = tpb
        self.__problem.bpg = bpg
        self.__tpb = tpb
        self.__bpg = (self.__rest_population // tpb[0] + 1,
                      self.__gene_size // tpb[0] + 1)

    def run(
            self,
            generations: int,
            verbose: bool = False,
            bar_style: str = "{l_bar}{bar:30}{r_bar}{bar:-30b}") -> None:
        # Create a progress bar
        progress_bar = tqdm(range(generations), bar_format=bar_style)

        for _ in progress_bar:
            self.step()

            # Update bar
            progress_bar.set_description(
                f"Value: {self.__best_value:.4f}")
            progress_bar.update()

        # Print info and results
        if verbose:
            title = Style.BRIGHT + Fore.LIGHTMAGENTA_EX
            print(Style.BRIGHT + '--------- INFO ---------')
            text = title + 'Population:\n' + Style.RESET_ALL
            text += f"  Total: {self.__population_size}\n"
            text += f"  Elites: {self.__elite_population}\n"
            text += f"  Mutants: {self.__mutants_population}"
            print(text)

            text = title + 'Best value:\n' + Style.RESET_ALL
            text += f"  {self.__best_value:.4f}"
            print(text)

            elapsed = progress_bar.format_dict['elapsed']
            text = title + 'Total time:\n' + Style.RESET_ALL
            text += f"  {float(elapsed):.4f} seconds"
            print(text)

    def step(self) -> None:
        # Decode current population
        decoded_population = self.__problem.decoder(
            self.__population,
            self.__population_size,
            self.__gene_size)

        # Local Search
        decoded_population = self.__problem.local_search(
            decoded_population,
            self.__info,
            self.__population_size,
            self.__gene_size
        )

        # Calculate fitness for each individual
        output = self.__problem.fitness(
            decoded_population,
            self.__info,
            self.__population_size,
            self.__gene_size
        )

        # Sort result
        output_index = cp.argsort(output)

        if self.__maximize:
            output_index = output_index[::-1]

        # Save best individual
        self.__best_value = output[output_index[0]]
        self.__best_individual = self.__population[output_index[0]]

        # Separate population in elites, commons and create the mutants
        elites = self.__population[output_index[:self.__elite_population]]
        commons = self.__population[output_index[self.__elite_population:]]
        mutants = cp.random.uniform(
            low=0, high=1,
            size=(self.__mutants_population, self.__gene_size),
            dtype=cp.float32)

        # Create next population
        next_population = cp.zeros(
            shape=(self.__population_size, self.__gene_size),
            dtype=cp.float32)

        # Copy current population to next
        ep = self.__elite_population
        mp = self.__mutants_population
        rp = self.__rest_population

        next_population[:ep, :] = elites
        next_population[ep:ep + mp, :] = mutants[:, :]

        # Generate random numbers necessary in crossover
        percentages = cp.random.uniform(
            low=0, high=1,
            size=(rp, self.__gene_size),
            dtype=cp.float32)

        output = cp.zeros((rp, self.__gene_size), dtype=cp.float32)

        # Process the indexes used in crossover
        elites_idx = cp.random.choice(elites.shape[0], rp, True)

        if not self.__mp:
            crossover_function = crossover
            commons_idx = cp.random.choice(commons.shape[0], rp, False)
        else:
            crossover_function = crossover_mp
            commons_idx = cp.concatenate((
                cp.random.choice(commons.shape[0], rp, False),
                cp.random.choice(commons.shape[0], rp, False)))

            # print(commons_idx.shape)
            # print(commons_idx.dtype)
            # exit(0)

        crossover_function(
                self.__bpg, self.__tpb,
                (percentages,
                 commons,
                 commons_idx,
                 elites,
                 elites_idx,
                 output,
                 cp.uint32(self.__gene_size),
                 cp.float32(self.__rhoe)))

        # Added the new commons from the crossover process to next population
        next_population[ep + mp:, :] = output
        self.__population = next_population
