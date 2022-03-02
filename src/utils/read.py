import cupy as cp
from typing import List, Tuple
import re


def read_cvrp_file(file_path: str) -> cp.ndarray:
    coord_section: bool = False
    demand_section: bool = False
    max_capacity: float = 0
    coords: List = []
    demands: List = []

    with open(file_path, 'r') as file:
        for line in file.readlines():
            text = line.strip()
            text = re.sub(' +', ' ', text)
            #
            if text == "NODE_COORD_SECTION":
                coord_section = True
                demand_section = False

            elif text == "DEMAND_SECTION":
                coord_section = False
                demand_section = True

            elif text == "DEPOT_SECTION":
                break

            elif "CAPACITY" in text:
                max_capacity = float(text.split(' ')[-1])

            elif coord_section:
                coords.append([float(x) for x in text.split(' ')])

            elif demand_section:
                demands.append([float(x) for x in text.split(' ')])

    coords = cp.array(coords, dtype=cp.float32)[:, 1:]
    demands = cp.array(demands, dtype=cp.float32)[:, 1:]
    infos = cp.concatenate((coords, demands), axis=1)

    return max_capacity, infos


def read_knapsack_file(file_path: str) -> Tuple[float, cp.ndarray]:
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
    infos = cp.dstack((weights, profits)).reshape(len(weights), 2)
    infos = cp.ascontiguousarray(infos)

    return max_weight, infos
