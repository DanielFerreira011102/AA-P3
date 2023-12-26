from math import ceil


class LossyCounter:
    def __init__(self, epsilon: float = 0.005):
        self.epsilon = epsilon
        self.width = ceil(1 / epsilon)
