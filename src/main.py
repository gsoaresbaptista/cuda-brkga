import sys
import os
import cupy as cp
import numpy as np
from brkga import BRKGA
from utils import read_knapsack_file
from knapsack import KnapsackProblem
from colorama import Fore, Style
from time import time


np.set_printoptions(precision=4)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise Exception('No input file provided.')

    if os.path.isfile(sys.argv[1]):
        weights, profits = read_knapsack_file(sys.argv[1])
        #
        weights = cp.array(weights, dtype=cp.float32)[1:]
        profits = cp.array(profits, dtype=cp.float32)[1:]
        info = cp.dstack((weights, profits)).reshape(weights.shape[0], 2)
        info = cp.ascontiguousarray(info)

        #
        try:
            problem = KnapsackProblem(20)
            text = Fore.GREEN + "Success: " + Style.RESET_ALL
            text += 'Loaded the knapsack problem!'
            print(text)
        except Exception:
            text = Fore.RED + "Error: " + Style.RESET_ALL
            text += 'Failed to load knapsack problem!'
            print(text)

        brkga = BRKGA(problem)
        brkga.fit_population(5000, 0.25, 0.10, 0.7)
        brkga.fit_input(info)
        start = time()
        brkga.run(1000)
        end = time()

        print('--------- INFO ---------')
        text = Fore.LIGHTMAGENTA_EX + 'Best value: ' + Style.RESET_ALL
        text += f"{brkga.best_value:.4f}"
        print(text)
        text = Fore.LIGHTMAGENTA_EX + 'Total time: ' + Style.RESET_ALL
        text += f"{end - start:.4f} seconds"
        print(text)

    else:
        raise Exception('File not exist.')
