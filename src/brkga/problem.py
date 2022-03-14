from typing import Tuple
import cupy as cp
from abc import ABC, abstractmethod
import os


class Problem(ABC):
    def __init__(self, file_path) -> None:
        self.__name = os.path.splitext(os.path.basename(file_path))[0]

    @property
    def name(self) -> str:
        return self.__name

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

    def local_search(
            self,
            population: cp.ndarray,
            info: cp.ndarray,
            population_size: int,
            gene_size: int,
            generation: int) -> cp.ndarray:
        return population

    def shaking(
            self,
            population: cp.ndarray,
            info: cp.ndarray,
            population_size: int,
            gene_size: int,
            generation: int) -> cp.ndarray:
        return population

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
