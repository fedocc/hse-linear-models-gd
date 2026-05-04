import numpy as np
from .interfaces import (
    LossFunction,
    LossFunctionClosedFormMixin,
    LinearRegressionInterface,
    AbstractOptimizer,
)
from .descents import AnalyticSolutionOptimizer
from typing import Dict, Type, Optional, Callable
from abc import abstractmethod, ABC
from scipy.sparse.linalg import svds


class MSELoss(LossFunction, LossFunctionClosedFormMixin):
    def __init__(
        self,
        analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
    ):

        if analytic_solution_func is None:
            self.analytic_solution_func = self._plain_analytic_solution
        else:
            self.analytic_solution_func = analytic_solution_func

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        X: np.ndarray, матрица регрессоров
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: float, значение MSE на данных X,y для весов w
        """
        r = X @ w - y
        return float(r @ r) / X.shape[0]

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: np.ndarray, численный градиент MSE в точке w
        """
        r = X @ w - y
        return (2 / X.shape[0]) * (X.T @ r)

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Возвращает решение по явной формуле (closed-form solution)

        X: np.ndarray, матрица регрессоров
        y: np.ndarray, вектор таргета

        returns: np.ndarray, оптимальный по MSE вектор весов, вычисленный при помощи аналитического решения для данных X, y
        """
        # Функция-диспатчер в одну из истинных функций для вычисления решения по явной формуле (closed-form)
        # Необходима в связи c наличием интерфейса analytic_solution у любого лосса;
        # self-injection даёт возможность выбирать, какое именно closed-form решение использовать
        return self.analytic_solution_func(X, y)

    @classmethod
    def _plain_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи классического аналитического решения
        """
        return np.linalg.inv(X.T @ X) @ X.T @ y

    @classmethod
    def _svd_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи аналитического решения на SVD
        """
        k = min(X.shape) - 1
        U, s, Vt = svds(X, k=k, solver="arpack", tol=1e-15)
        idx = np.argsort(s)[::-1]
        s = s[idx]
        U = U[:, idx]
        Vt = Vt[idx, :]
        eps = 1e-15
        s_inv = np.where(s > eps, 1 / s, 0)
        return Vt.T @ (s_inv * (U.T @ y))


class L2Regularization(LossFunction):
    def __init__(
        self,
        core_loss: LossFunction,
        mu_rate: float = 1.0,
        analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
        regularize_bias: bool = False,
    ):
        self.core_loss = core_loss
        self.mu_rate = mu_rate
        self.regularize_bias = regularize_bias

        # analytic_solution_func is meant to be passed separately,
        # as it is not linear to core solution

    def _weights_for_penalty(self, w: np.ndarray) -> np.ndarray:
        if self.regularize_bias or w.size == 0:
            return w
        w_reg = w.copy()
        w_reg[-1] = 0.0
        return w_reg

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        core_part = self.core_loss.loss(X, y, w)
        w_reg = self._weights_for_penalty(w)
        reg_term = (self.mu_rate / 2) * (w_reg @ w_reg)
        return core_part + reg_term

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        core_part = self.core_loss.gradient(X, y, w)
        w_reg = self._weights_for_penalty(w)
        reg_term = self.mu_rate * w_reg
        return core_part + reg_term

class LogCoshLoss(LossFunction):
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        diff = X @ w - y
        return np.mean(np.log(np.cosh(diff)))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        diff = X @ w - y
        return (1 / n) * (X.T @ np.tanh(diff))

class HuberLoss(LossFunction):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        error = X @ w - y
        abs_error = np.abs(error)
        quadratic = 0.5 * (error**2)
        linear = self.delta * abs_error - 0.5 * (self.delta**2)
        return np.mean(np.where(abs_error <= self.delta, quadratic, linear))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        error = X @ w - y
        abs_error = np.abs(error)
        
        grad_values = np.where(
            abs_error <= self.delta, 
            error, 
            self.delta * np.sign(error)
        )
        return (1 / n) * (X.T @ grad_values)

class CustomLinearRegression(LinearRegressionInterface):
    def __init__(
        self,
        optimizer: AbstractOptimizer,
        # l2_coef: float = 0.0,
        loss_function: LossFunction = MSELoss(),
    ):
        self.optimizer = optimizer
        self.optimizer.set_model(self)

        # self.l2_coef = l2_coef
        self.loss_function = loss_function
        self.loss_history = []
        self.w = None
        self.X_train = None
        self.y_train = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        returns: np.ndarray, вектор y_hat
        """
        return X @ self.w

    def compute_gradients(
        self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None
    ) -> np.ndarray:
        """
        returns: np.ndarray, градиент функции потерь при текущих весах (self.w)
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        if X_batch is None and y_batch is None:
            return self.loss_function.gradient(self.X_train, self.y_train, self.w)
        return self.loss_function.gradient(X_batch, y_batch, self.w)

    def compute_loss(
        self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None
    ) -> float:
        """
        returns: np.ndarray, значение функции потерь при текущих весах (self.w) по self.X_train, self.y_train
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        if X_batch is None and y_batch is None:
            return self.loss_function.loss(self.X_train, self.y_train, self.w)
        return self.loss_function.loss(X_batch, y_batch, self.w)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Инициирует обучение модели заданным функцией потерь и оптимизатором способом.

        X: np.ndarray,
        y: np.ndarray
        """
        self.X_train, self.y_train = X, y
        self.optimizer.optimize()
