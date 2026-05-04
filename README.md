# HSE Linear Models and Gradient Descent

Аккуратно оформленная версия моей домашней работы по курсу машинного обучения НИУ ВШЭ: линейная регрессия и методы градиентной оптимизации с нуля.

Основная идея проекта — реализовать линейную регрессию не через `sklearn.fit`, а как набор независимых компонентов: функции потерь, оптимизаторы, learning rate schedules, регуляризация и эксперименты на реальном табличном датасете.

## Что внутри

- closed-form решение MSE;
- SVD-based решение для случая мультиколлинеарности;
- собственная `CustomLinearRegression`;
- Vanilla GD, SGD, SAG, Momentum, Adam;
- L2-регуляризация;
- LogCosh и Huber loss;
- сравнение оптимизаторов на задаче предсказания цены автомобиля.

## Результаты

### Closed-form solution

| Проверка | MSE sklearn | MSE custom |
|---|---:|---:|
| Обычная матрица признаков | 0.1064 | 0.1064 |
| Мультиколлинеарность, naive inverse | 0.1095 | 0.1187 |
| Мультиколлинеарность, SVD solution | 0.1095 | 0.1095 |

### Сравнение оптимизаторов

| Optimizer | Test MSE | Test R2 | Iterations |
|---|---:|---:|---:|
| Vanilla GD | 0.2039 | 0.8235 | 389 |
| SGD | 0.2266 | 0.8039 | 1000 |
| SAG | 0.2086 | 0.8195 | 485 |
| Momentum | 0.1895 | 0.8359 | 634 |
| Adam | 0.1864 | 0.8386 | 1000 |

## Структура

```text
.
├── data/
│   └── README.md
├── docs/
│   └── experiment_summary.md
├── notebooks/
│   └── linear_models_and_gd_experiments.ipynb
├── src/
│   └── linear_models_gd/
│       ├── __init__.py
│       ├── descents.py
│       ├── interfaces.py
│       └── linear_regression.py
└── requirements.txt
```

## Данные

Сырые данные не лежат в репозитории. Ожидаемая локальная структура:

```text
data/raw/autos.csv
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Быстрая проверка

```bash
PYTHONPATH=src python - <<'PY'
import numpy as np
from linear_models_gd import CustomLinearRegression, MSELoss, VanillaGradientDescent, ConstantLR

X = np.random.default_rng(42).normal(size=(100, 5))
w_true = np.array([1, -2, 0.5, 0, 3])
y = X @ w_true

model = CustomLinearRegression(
    optimizer=VanillaGradientDescent(lr_schedule=ConstantLR(0.05), max_iter=500),
    loss_function=MSELoss(),
)
model.fit(X, y)
print(model.compute_loss())
PY
```

## Заметки

Notebook содержит графики, таблицы и метрики по основным экспериментам.
