import cupy as cp
from typing import Tuple


def read_cvrp_file(file_path: str) -> Tuple[cp.ndarray, cp.ndarray]:
    coord_section = False
    demand_section = False
    coords = []
    demands = []

    with open(file_path, 'r') as file:
        for line in file.readlines():
            text = line.strip()
            #
            if text == "NODE_COORD_SECTION":
                coord_section = True
                demand_section = False

            elif text == "DEMAND_SECTION":
                coord_section = False
                demand_section = True

            elif text == "DEPOT_SECTION":
                break

            elif coord_section:
                coords.append([float(x) for x in text.split(' ')])

            elif demand_section:
                demands.append([float(x) for x in text.split(' ')])

    return coords, demands


def read_knapsack_file(file_path: str) -> Tuple[cp.ndarray, cp.ndarray]:
    weights = []
    profits = []

    with open(file_path, 'r') as file:
        for line in file.readlines():
            profit, weight = line.strip().split(' ')
            weights.append(float(weight))
            profits.append(float(profit))

    max_weight = weights[0]
    weights = cp.array(weights, dtype=cp.float32)[1:]
    profits = cp.array(profits, dtype=cp.float32)[1:]
    infos = cp.dstack((weights, profits)).reshape(weights.shape[0], 2)
    infos = cp.ascontiguousarray(infos)

    return max_weight, infos
