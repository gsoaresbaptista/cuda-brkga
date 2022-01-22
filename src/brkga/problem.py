from typing import Tuple
import cupy as cp
from abc import ABC, abstractmethod


class Problem(ABC):
    def __init__(self) -> None:
        ...

    @property
    def tpb(self) -> Tuple[int, int]:
        return self.__tpb

    @tpb.setter
    def tpb(self, new: Tuple[int, int]) -> None:
        self.__tpb = new

    @property
    def bpg(self) -> Tuple[int, int]:
        return self.__bpg

    @bpg.setter
    def bpg(self, new: Tuple[int, int]) -> None:
        self.__bpg = new

    @abstractmethod
    def decoder(
            self,
            population: cp.ndarray,
            population_size: int,
            gene_size: int) -> cp.ndarray:
        ...

    @abstractmethod
    def fitness(
            self,
            population: cp.ndarray,
            info: cp.ndarray,
            population_size: int,
            gene_size: int) -> cp.ndarray:
        ...
