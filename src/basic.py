from abc import ABCMeta, abstractmethod


class Solution(metaclass=ABCMeta):
    def __init__(self, name: str):
        self.name = name
        self.input_data = []
        self.ans = ""

    @abstractmethod
    def read_input(self, input: str) -> "Solution":
        """
        @param input: 非多组输入，我们默认 `input` 是输入格式的输入，换行以 `\n` 分割；多组输入，我们默认 `input` 是输入文件的名称，需要读取文件内容
        """
        return self

    @abstractmethod
    def solve(self) -> str:
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
