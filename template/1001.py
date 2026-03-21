class Solution:
    def __init__(self):
        pass

    def solve(self, input: str) -> str:
        return bin(int(input))[2:]


if __name__ == "__main__":
    solution = Solution()
    print(solution.solve(input()))
