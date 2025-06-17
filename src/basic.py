from abc import ABC, abstractmethod


class Solution(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def solve(self, input: str) -> str:
        pass

class Generator:

    def problem_description(self, path: str, solution: Solution):
        with open(path, "w") as f:
            lines = solution.__doc__
            f.write(lines)
            f.write("\n")


    @abstractmethod
    def generate_data(self, solution: Solution):
        pass
