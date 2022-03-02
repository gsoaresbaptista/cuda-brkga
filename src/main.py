import os
import argparse
from brkga import BRKGA
from brkga.problem import Problem
from cvrp import CVRPProblem
from utils import read_knapsack_file, read_cvrp_file
from knapsack import KnapsackProblem
from colorama import Fore, Style


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", choices=["knapsack", "cvrp"], type=str)
    parser.add_argument("input", type=str)
    parser.add_argument("-g", "--generations", default=300, type=int)
    parser.add_argument("-p", "--population_size", default=1000, type=int)
    parser.add_argument("-pe", "--elite_percentage", default=0.2, type=float)
    parser.add_argument("-pm", "--mutants_percentage", default=0.1, type=float)
    parser.add_argument("-re", "--rhoe", default=0.7, type=float)
    parser.add_argument("-mp", "--multiparent", type=bool, default=False)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if os.path.isfile(args.input):
        problem: Problem

        if args.problem == 'knapsack':
            max_weight, infos = read_knapsack_file(args.input)
            problem = KnapsackProblem(max_weight)
            maximize = True
            gene_size = infos.shape[0]
        elif args.problem == 'cvrp':
            max_capacity, infos = read_cvrp_file(args.input)
            problem = CVRPProblem(max_capacity)
            maximize = False
            gene_size = infos.shape[0] - 1

        try:
            text = Fore.GREEN + "Success: " + Style.RESET_ALL
            text += f'Loaded the {problem.__class__.__name__}!'
            print(text)
        except Exception:
            text = Fore.RED + "Error: " + Style.RESET_ALL
            text += f'Failed to load {problem.__class__.__name__}!'
            print(text)

        brkga = BRKGA(problem, gene_size, args.multiparent, maximize)

        if args.seed is not None:
            brkga.set_seed(args.seed)

        brkga.fit_population(
            args.population_size,
            args.elite_percentage,
            args.mutants_percentage,
            args.rhoe)
        brkga.fit_input(infos)
        brkga.run(args.generations, verbose=True)
    else:
        raise Exception('File not exist.')
