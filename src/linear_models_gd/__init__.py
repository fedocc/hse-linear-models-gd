"""Linear regression and gradient descent implementations for HSE ML coursework."""

from .descents import (
    Adam,
    AnalyticSolutionOptimizer,
    ConstantLR,
    MomentumDescent,
    SAGDescent,
    StochasticGradientDescent,
    TimeDecayLR,
    VanillaGradientDescent,
)
from .linear_regression import (
    CustomLinearRegression,
    HuberLoss,
    L2Regularization,
    LogCoshLoss,
    MSELoss,
)

__all__ = [
    "Adam",
    "AnalyticSolutionOptimizer",
    "ConstantLR",
    "MomentumDescent",
    "SAGDescent",
    "StochasticGradientDescent",
    "TimeDecayLR",
    "VanillaGradientDescent",
    "CustomLinearRegression",
    "HuberLoss",
    "L2Regularization",
    "LogCoshLoss",
    "MSELoss",
]

