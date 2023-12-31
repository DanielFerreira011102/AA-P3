from counter import Counter
from permetrics import RegressionMetric
from evaluator import CounterEvaluator

def main():
    d1 = {'O': 20431, 'R': 15341, 'T': 28708, 'H': 16782, 'D': 9098, 'X': 605, 'Y': 5848, 'B': 4453, 'G': 5164, 'I': 22738, 'L': 12025, 'E': 33535, 'K': 1702, 'C': 8019, 'S': 18860, 'N': 19659, 'P': 4648, 'F': 6252, 'A': 23528, 'M': 7351, 'U': 7541, 'V': 2784, 'W': 5254, 'Z': 113, 'Q': 253, 'J': 293, '1': 2, '8': 2, '7': 2, '0': 2}
    d2 = {'N': 19720, 'D': 8800, 'F': 6240, 'R': 15200, 'S': 18920, 'E': 33200, 'G': 5040, 'I': 22720, 'H': 16760, 'U': 7480, 'C': 7880, 'T': 28840, 'P': 4320, 'L': 11960, 'V': 2680, 'A': 24160, 'X': 600, 'Y': 6000, 'O': 20080, 'M': 7520, 'B': 4480, 'W': 5120, 'K': 1760, 'J': 280, 'Q': 280, 'Z': 80, '1': 0}
    d1 = dict(sorted(d1.items(), key=lambda item: item[0]))
    d2 = dict(sorted(d2.items(), key=lambda item: item[0]))
    print(d1)
    print(d2)
    c1 = Counter("f")
    c1.assign(d1)
    c2 = Counter("f")
    c2.assign(d2)
    evalu = CounterEvaluator(c1, c2)

    print(evalu.correlation())


if __name__ == "__main__":
    main()