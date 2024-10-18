import numpy as np


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class QuadraticRunningObjective:
    def __init__(self, weights: np.ndarray, biases: np.ndarray | float = 0.0) -> None:
        self.weights = weights
        self.biases = biases

    def __call__(self, state: np.ndarray, action: np.ndarray) -> float:
        return (np.hstack((state - self.biases, action)) ** 2 * self.weights).sum()


class GymPendulumRunningObjective:

    def __call__(self, state, action):
        return (
            angle_normalize(state[0]) ** 2
            + 0.1 * state[1] ** 2
            + 0.001 * action[0] ** 2
        )


class PendulumRunningObjective:
    def __init__(self) -> None:
        pass

    def __call__(self, state: np.ndarray, action: np.ndarray):
        return (1 - np.cos(state[0])) ** 2 + state[1] ** 2
        # return (
        #     angle_normalize(state[0]) ** 2
        #     + 0.1 * state[1] ** 2
        #     + 0.001 * action[0] ** 2
        # )
