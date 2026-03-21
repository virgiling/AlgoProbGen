class Solution:
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

    def solve(self, input: str) -> str:
        base, n, target_base = input.split("\n")
        return self._to_base_n(self._to_base_10(n, int(base)), int(target_base))

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


if __name__ == "__main__":
    solution = Solution()
    print(solution.solve(input()))
