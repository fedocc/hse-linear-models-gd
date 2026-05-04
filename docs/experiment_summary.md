# Сводка экспериментов

Репозиторий содержит реализацию линейной регрессии и нескольких методов градиентной оптимизации с нуля.

## Реализованные компоненты

- `MSELoss`;
- `L2Regularization`;
- `LogCoshLoss`;
- `HuberLoss`;
- `VanillaGradientDescent`;
- `StochasticGradientDescent`;
- `SAGDescent`;
- `MomentumDescent`;
- `Adam`;
- `ConstantLR` и `TimeDecayLR`;
- closed-form solution через обычную формулу и SVD.

## Проверка closed-form solution

| Проверка | MSE sklearn | MSE custom |
|---|---:|---:|
| Обычная матрица признаков | 0.1064 | 0.1064 |
| Мультиколлинеарность, naive inverse | 0.1095 | 0.1187 |
| Мультиколлинеарность, SVD solution | 0.1095 | 0.1095 |

Вывод: SVD-решение устойчивее обычного обращения матрицы в случае мультиколлинеарности.

## Данные автомобилей

| Объект | Значение |
|---|---:|
| Raw rows | 241,190 |
| Rows after cleaning | 228,919 |
| Target | `log_price` |
| Feature matrix after preprocessing | 183,135 x 307 |

Основные шаги preprocessing:

- логарифмирование `price` и `powerPS`;
- удаление выбросов через IQR;
- train/validation/test split до fitting encoders/scalers;
- OneHotEncoding категориальных признаков;
- StandardScaler числовых признаков;
- bias column.

## Сравнение оптимизаторов

| Optimizer | Test MSE | Test R2 | Iterations |
|---|---:|---:|---:|
| Vanilla GD | 0.2039 | 0.8235 | 389 |
| SGD | 0.2266 | 0.8039 | 1000 |
| SAG | 0.2086 | 0.8195 | 485 |
| Momentum | 0.1895 | 0.8359 | 634 |
| Adam | 0.1864 | 0.8386 | 1000 |

Лучшие результаты в экспериментах дали Adam и Momentum.

## Основные выводы

1. SVD closed-form solution нужен для устойчивости при вырожденной матрице признаков.
2. Adam и Momentum дают лучшие результаты на подготовленной задаче регрессии.
3. SGD сильно зависит от batch size: маленькие батчи дают шумный градиент, большие приближают метод к full-batch.
4. L2-регуляризация стабилизирует обучение и снижает риск взрыва весов.
5. Huber и LogCosh делают оптимизацию спокойнее на данных с выбросами.

