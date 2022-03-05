import cupy as cp
from typing import Callable, List
from tqdm.autonotebook import tqdm
from .problem import Problem
from .kernel import crossover, crossover_mp, shuffle_params
from colorama import Fore, Style


class BRKGA:
    def __init__(
            self,
            problem: Problem,
            gene_size: int,
            Ip: int,
            Ii: int,
            Ig: int,
            mp: bool = False,
            maximize: bool = True) -> None:
        self.__problem = problem
        self.__population_size = 0
        self.__elite_population = 0
        self.__mutants_population = 0
        self.__rest_population = 0
        self.__rhoe = 0.0
        self.__info: cp.ndarray = None
        self.__population: cp.ndarray = None
        self.__maximize = maximize
        self.__comparision_solution = self.__update_best()
        self.__tpb = (0, 0)
        self.__bpg = (0, 0)
        self.__best_value = 0
        self.__best_individual = None
        self.__gene_size = gene_size
        self.__mp = mp
        self.__Ip = Ip
        self.__Ii = Ii
        self.__Ig = Ig

    def __update_best(self) -> Callable:
        if self.__maximize:
            return (lambda x: self.__best_value == 0 or x > self.__best_value)
        else:
            return (lambda x: self.__best_value == 0 or x < self.__best_value)

    def __phi_function(self, ranks: cp.ndarray) -> cp.ndarray:
        return 1/(ranks)
        # return cp.exp(ranks)**-1

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
            rhoe: float,
            pi_total: int,
            pi_elite: int) -> None:
        # Parameters
        self.__rhoe = rhoe
        self.__population_size = p
        self.__elite_population = int(p * pe)
        self.__mutants_population = int(p * pm)
        self.__rest_population = int(p * (1.0 - pe - pm))
        self.__pi_total = pi_total
        self.__pi_elite = pi_elite

    def fit_input(self, info: List) -> None:
        if self.__rhoe == 0.0:
            raise Exception(
                "Set population parameters before fitting to input.")

        self.__info = cp.array(info, dtype=cp.float32)
        self.__population = cp.random.uniform(
            low=0, high=1,
            size=(self.__Ip, self.__population_size, self.__gene_size),
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
        #
        self.__apply_local_search = 0.05 * generations

        # Create a progress bar
        progress_bar = tqdm(range(generations), bar_format=bar_style)

        for generation, _ in enumerate(progress_bar):
            for i in range(self.__Ip):
                self.step(i, generation)

            if (self.__Ip > 1) and (generation != 0) and \
               (generation % self.__Ig == 0):
                # Process migration to the left island
                for i in range(self.__Ip):
                    if i == self.__Ip - 1:
                        self.__population[i, -self.__Ii:, :] = \
                            self.__population[0, :self.__Ii, :]
                    else:
                        self.__population[i, -self.__Ii:, :] = \
                            self.__population[i + 1, :self.__Ii, :]

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

    def step(self, population_id: int, generation: int) -> None:
        # Decode current population
        decoded_population = self.__problem.decoder(
            self.__population[population_id],
            self.__population_size,
            self.__gene_size)

        # Local Search
        decoded_population = self.__problem.local_search(
            decoded_population,
            self.__info,
            self.__population_size,
            self.__gene_size,
            generation,
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
        if self.__comparision_solution(output[output_index[0]]):
            self.__best_value = output[output_index[0]]
            self.__best_individual = \
                self.__population[population_id][output_index[0]]

        # Separate population in elites, commons and create the mutants
        elites = self.__population[
            population_id, output_index[:self.__elite_population]]
        commons = self.__population[
            population_id, output_index[self.__elite_population:]]
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

        if not self.__mp:
            # Crossover without multi-parent
            elites_idx = cp.random.choice(elites.shape[0], rp, True)
            commons_idx = cp.random.choice(commons.shape[0], rp, False)
            output = cp.zeros((rp, self.__gene_size), dtype=cp.float32)

            crossover(
                self.__bpg, self.__tpb,
                (percentages,
                 commons,
                 commons_idx,
                 elites,
                 elites_idx,
                 output,
                 cp.uint32(self.__gene_size),
                 cp.float32(self.__rhoe)))
        else:
            # Crossover with multi-parent
            output = cp.zeros((rp, self.__gene_size), dtype=cp.float32)

            # Random choice of parents
            chosen_elites = cp.random.choice(
                elites.shape[0], (rp, self.__pi_elite))
            chosen_commons = cp.random.choice(
                commons.shape[0], (rp, self.__pi_total - self.__pi_elite))

            # Calculate individual ranks
            chosen_elites_ranks = chosen_elites + 1
            chosen_commons_ranks = chosen_commons + self.__elite_population
            ranks = cp.concatenate((
                chosen_elites_ranks, chosen_commons_ranks), axis=1)
            weight = self.__phi_function(ranks)
            weight = cp.divide(weight.T, cp.sum(weight, axis=1)) \
                       .T.astype(cp.float32)

            # Create output arrays of shuffle process
            indices = cp.concatenate((chosen_elites, chosen_commons_ranks),
                                     axis=1).astype(cp.uint32)
            c_indices = cp.zeros(indices.shape, dtype=cp.uint32)
            c_weights = cp.zeros(weight.shape, dtype=cp.float32)
            p = cp.random.randint(0, self.__pi_total,
                                  (rp, self.__pi_total), dtype=cp.uint32)

            # Shuffle the parameters for better parent selection
            shuffle_params((rp,), (1,), (indices, c_indices,
                           weight, c_weights, p, cp.uint32(self.__pi_total)))

            # Process multi-parent crossover
            crossover_mp((rp,), (1,), (
                c_indices, c_weights,
                cp.random.random((rp, self.__gene_size), dtype=cp.float32),
                self.__population[population_id][output_index],
                output, cp.uint32(self.__gene_size),
                cp.uint32(self.__pi_total)))

        # Added the new commons from the crossover process to next population
        next_population[ep + mp:, :] = output
        self.__population[population_id] = next_population
