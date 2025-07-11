import os
import tempfile
from basic import Generator, Solution
import unittest
import cyaron
import argparse


# 单组输入例子


class SingleInputCase(Solution):
    """
    # 题目描述

    请你编一程序实现两种不同进制之间的数据转换。

    # 输入格式

    共三行，第一行是一个正整数，表示需要转换的数的进制 n ($2 \leq n \leq 16$)，第二行是一个 n 进制数，若 $n \gt 10$ 则用大写字母 A~F 表示数码 10~15，并且该 n 进制数对应的十进制的值不超过 $10^9$，第三行也是一个正整数，表示转换之后的数的进制 m ($2 \leq m \leq 16$)。

    # 输出格式

    一个正整数，表示转换之后的 m 进制数。

    # 输入样例

    ```txt
    16
    FF
    2
    ```

    # 输出样例

    ```txt
    11111111
    ```
    """

    def __init__(self):
        self.base_map = {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10: "A",
            11: "B",
            12: "C",
            13: "D",
            14: "E",
            15: "F",
        }
        super().__init__("SingleInputCase")

    def read_input(self, input: str):
        self.input_data.append(input)
        return self

    def solve(self) -> str:
        self.ans = ""
        for line in self.input_data:
            base, n, target_base = line.split("\n")
            self.ans += self._to_base_n(
                self._to_base_10(n, int(base)), int(target_base)
            )
        return self.ans

    def _to_base_10(self, n: str, base: int) -> int:
        num = 0
        for c in n:
            tmp = ord(c) - ord("0") if c.isdigit() else ord(c) - ord("A") + 10
            num = num * base + int(tmp)
        return num

    def _to_base_n(self, n: int, base: int) -> str:
        num = ""
        while n > 0:
            num = self.base_map[n % base] + num
            n = n // base
        return num


# 多组输入例子


class MultipleInputCase(Solution):
    """
    # 题目描述

    将一个长度最多为 128 位数字的十进制非负整数转换为二进制数输出。

    # 输入格式

    有多组输入，每组输入一行，是一个不超过 128 位的正整数。

    # 输出格式

    对于每组输入，输出一行，一个二进制数，表示转换之后的二进制数。

    # 输入样例

    ```txt
    985
    1234567890123456789012345678901234567890
    1000
    ```

    # 输出样例

    ```txt
    1111011001
    1110100000110010010010000001110101110000001101101111110011101110001010110010111100010111111001011011001110001111110000101011010010
    1111101000
    ```
    """

    def __init__(self):
        super().__init__("MultipleInputCase")

    def read_input(self, input: str):
        with open(input, "r") as f:
            file_content = f.readlines()
            for line in file_content:
                self.input_data.append(line.strip())
        return self

    def solve(self) -> str:
        self.ans = ""
        for line in self.input_data:
            self.ans += bin(int(line))[2:] + "\n"
        return self.ans


class SampleTest(unittest.TestCase):
    def test_SingleInputCase(self):
        prob = SingleInputCase().read_input("16\nFF\n2")
        self.assertEqual(prob.solve(), "11111111")

    def test_MultipleInputCase(self):
        fd, filename = tempfile.mkstemp()
        os.write(fd, b"985\n1234567890123456789012345678901234567890\n1000")
        os.close(fd)
        prob = MultipleInputCase().read_input(filename)
        self.assertEqual(
            prob.solve(),
            "1111011001\n1110100000110010010010000001110101110000001101101111110011101110001010110010111100010111111001011011001110001111110000101011010010\n1111101000\n",
        )


class ProblemGenerator(Generator):
    def generate_data(self, solution: Solution):
        if isinstance(solution, SingleInputCase):
            os.makedirs(f"data/{solution.name}/data", exist_ok=True)
            for i in range(1, 6):
                path = f"data/{solution.name}/data/{i}.in"
                base = cyaron.randint(2, 16)
                new_base = cyaron.randint(2, 16)
                num = cyaron.randint(1, 10**9)
                num_str = ""
                while num > 0:
                    num_str = solution.base_map[num % base] + num_str
                    num = num // base
                with open(path, "w") as f:
                    f.write(f"{base}\n{num_str}\n{new_base}\n")
                with open(path.replace(".in", ".out"), "w") as f:
                    f.write(solution.solve(f"{base}\n{num_str}\n{new_base}"))

        elif isinstance(solution, MultipleInputCase):
            os.makedirs(f"data/{solution.name}/data", exist_ok=True)
            for i in range(1, 6):
                path = f"data/{solution.name}/data/{i}.in"
                num = cyaron.String.random((10, 128), charset="0123456789")
                with open(path, "w") as f:
                    f.write(num)
                with open(path.replace(".in", ".out"), "w") as f:
                    f.write(solution.solve(num))
        else:
            raise ValueError(f"Unknown Problem: {solution.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    if args.test:
        unittest.main()
    elif args.generate:
        gen = ProblemGenerator()
        problem_class_name = [
            "SingleInputCase",
            "MultipleInputCase",
        ]
        for problem in problem_class_name:
            solution = globals()[problem]()
            os.makedirs(f"data/{solution.name}", exist_ok=True)
            gen.problem_description(f"data/{solution.name}/description.md", solution)
            gen.generate_data(solution)
