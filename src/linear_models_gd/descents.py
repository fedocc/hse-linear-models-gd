import numpy as np
from abc import ABC, abstractmethod
from .interfaces import (
    LearningRateSchedule,
    AbstractOptimizer,
    LinearRegressionInterface,
)


# ===== Learning Rate Schedules =====
class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        """
        returns: float, learning rate для iteration шага обучения
        """
        return self.lambda_ * (self.s0 / (self.s0 + iteration)) ** self.p


# ===== Base Optimizer =====
class BaseDescent(AbstractOptimizer, ABC):
    """
    Оптимизатор, имплементирующий градиентный спуск.
    Ответственен только за имплементацию общего алгоритма спуска.
    Все его составные части (learning rate, loss function+regularization) находятся вне зоны ответственности этого класса (см. Single Responsibility Principle).
    """

    def __init__(
        self,
        lr_schedule: LearningRateSchedule = TimeDecayLR(),
        tolerance: float = 1e-6,
        max_iter: int = 1000,
    ):
        self.lr_schedule = lr_schedule
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.iteration = 0
        self.model: LinearRegressionInterface = None

    @abstractmethod
    def _update_weights(self) -> np.ndarray:
        """
        Вычисляет обновление согласно конкретному алгоритму и обновляет веса модели, перезаписывая её атрибут.
        Не имеет прямого доступа к вычислению градиента в точке, для подсчета вызывает model.compute_gradients.

        returns: np.ndarray, w_{k+1} - w_k
        """
        pass

    def _step(self) -> np.ndarray:
        """
        Проводит один полный шаг интеративного алгоритма градиентного спуска

        returns: np.ndarray, w_{k+1} - w_k
        """
        delta = self._update_weights()
        self.iteration += 1
        return delta

    def optimize(self) -> None:
        """
        Оркестрирует весь алгоритм градиентного спуска.
        """
        self.iteration = 0
        self.model.loss_history = []
        self.model.w = np.zeros(self.model.X_train.shape[1])
        self.model.loss_history.append(self.model.compute_loss())
        for _ in range(self.max_iter):
            delta = self._step()
            if (
                np.isnan(delta).any()
                or np.isinf(delta).any()
                or np.isnan(self.model.w).any()
                or np.isinf(self.model.w).any()
            ):
                self.model.loss_history.append(np.nan)
                break
            if delta @ delta < self.tolerance:
                self.model.loss_history.append(self.model.compute_loss())
                break
            self.model.loss_history.append(self.model.compute_loss())


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        w_prev = self.model.w
        self.model.w = w_prev - self.lr_schedule.get_lr(
            self.iteration
        ) * self.model.compute_gradients(X_train, y_train)
        return self.model.w - w_prev


class StochasticGradientDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        n = self.model.X_train.shape[0]
        idx = np.random.randint(0, n, size=min(self.batch_size, n))
        X_batch = self.model.X_train[idx]
        y_batch = self.model.y_train[idx]
        w_prev = self.model.w
        self.model.w = w_prev - self.lr_schedule.get_lr(
            self.iteration
        ) * self.model.compute_gradients(X_batch, y_batch)
        return self.model.w - w_prev


class SAGDescent(BaseDescent):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_memory = None
        self.grad_sum = None
        self.avg_grad = None
        self.batch_size = batch_size

    def _update_weights(self) -> np.ndarray:
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.iteration == 0:
            self.grad_memory = np.zeros((num_objects, num_features), dtype=float)
            self.avg_grad = np.zeros(num_features, dtype=float)

        idx = np.random.choice(
            num_objects, size=min(self.batch_size, num_objects), replace=False
        )
        old_grads = self.grad_memory[idx]
        new_grads = np.empty_like(old_grads)
        for i, j in enumerate(idx):
            new_grads[i] = self.model.compute_gradients(
                X_train[j : j + 1], y_train[j : j + 1]
            )
        self.grad_memory[idx] = new_grads
        self.avg_grad += (new_grads - old_grads).sum(axis=0) / num_objects
        w_prev = self.model.w
        self.model.w = w_prev - self.lr_schedule.get_lr(self.iteration) * self.avg_grad
        return self.model.w - w_prev


class MomentumDescent(BaseDescent):
    def __init__(self, *args, beta=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.velocity = None

    def _update_weights(self) -> np.ndarray:
        if self.iteration == 0:
            self.velocity = np.zeros_like(self.model.w)

        self.velocity = (
            self.beta * self.velocity
            + self.lr_schedule.get_lr(self.iteration) * self.model.compute_gradients()
        )
        w_prev = self.model.w
        self.model.w = w_prev - self.velocity
        return self.model.w - w_prev


class Adam(BaseDescent):
    def __init__(self, *args, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def _update_weights(self) -> np.ndarray:
        if self.iteration == 0:
            self.m = np.zeros_like(self.model.w)
            self.v = np.zeros_like(self.model.w)

        grad = self.model.compute_gradients()
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        m_hat = self.m / (1 - self.beta1 ** (self.iteration + 1))
        v_hat = self.v / (1 - self.beta2 ** (self.iteration + 1))

        w_prev = self.model.w
        self.model.w = w_prev - self.lr_schedule.get_lr(self.iteration) * m_hat / (
            np.sqrt(v_hat) + self.eps
        )
        return self.model.w - w_prev


# ===== Non-iterative Algorithms ====
class AnalyticSolutionOptimizer(AbstractOptimizer):
    """
    Универсальный дамми-класс для вызова аналитических решений
    """

    def __init__(self):
        self.model = None

    def optimize(self) -> None:
        """
        Определяет аналитическое решение и назначает его весам модели.
        """
        # не должна содержать непосредственных формул аналитического решения, за него ответственен другой объект
        self.model.w = self.model.loss_function.analytic_solution(
            self.model.X_train, self.model.y_train
        )
