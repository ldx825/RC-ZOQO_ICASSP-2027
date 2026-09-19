from __future__ import annotations

import numpy as np


class QueryObjective:
    def __init__(self) -> None:
        self.query_count = 0

    def __call__(self, x: np.ndarray) -> float:
        self.query_count += 1
        return self.value(np.asarray(x, dtype=np.float64))

    def value(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def reset_queries(self) -> None:
        self.query_count = 0


class QuadraticObjective(QueryObjective):
    def __init__(self, A: np.ndarray, x_star: np.ndarray):
        super().__init__()
        self.A = np.asarray(A, dtype=np.float64)
        self.x_star = np.asarray(x_star, dtype=np.float64)

    def value(self, x: np.ndarray) -> float:
        error = x - self.x_star
        return float(0.5 * np.sum(self.A * error * error)) if self.A.ndim == 1 else float(0.5 * error @ self.A @ error)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        error = np.asarray(x) - self.x_star
        return self.A * error if self.A.ndim == 1 else self.A @ error

    @property
    def smoothness(self) -> float:
        return float(self.A.max() if self.A.ndim == 1 else np.linalg.eigvalsh(self.A).max())


class LinearObjective(QueryObjective):
    def __init__(self, coefficient: np.ndarray):
        super().__init__()
        self.coefficient = np.asarray(coefficient, dtype=np.float64)

    def value(self, x: np.ndarray) -> float:
        return float(self.coefficient @ x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.coefficient.copy()


class LogisticObjective(QueryObjective):
    def __init__(self, X: np.ndarray, y: np.ndarray, l2: float = 1e-2):
        super().__init__()
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.l2 = float(l2)

    def value(self, w: np.ndarray) -> float:
        margins = self.y * (self.X @ w)
        return float(np.logaddexp(0.0, -margins).mean() + 0.5 * self.l2 * (w @ w))

    def gradient(self, w: np.ndarray) -> np.ndarray:
        margins = self.y * (self.X @ w)
        weights = -self.y / (1.0 + np.exp(np.clip(margins, -50, 50)))
        return self.X.T @ weights / self.X.shape[0] + self.l2 * w

    def accuracy(self, w: np.ndarray) -> float:
        return float(np.mean(np.where(self.X @ w >= 0, 1.0, -1.0) == self.y))

