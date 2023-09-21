import random
from math import sqrt

import pandas as pd


def gerar_C1(n_samples: int):
    points = []
    for _ in range(n_samples):
        x = random.uniform(0, 1)
        y = random.uniform(0, -x + 1)
        points.append([x, y, 0])

    return points


def gerar_C2(n_samples: int):
    points = []
    for _ in range(n_samples):
        x = random.uniform(-1, 0)
        y = random.uniform(0, x + 1)
        points.append([x, y, 1])

    return points


def gerar_C3(n_samples: int):
    points = []
    for _ in range(n_samples):
        x = random.uniform(-1, 0)
        y = random.uniform(-1 - x, 0)
        points.append([x, y, 2])

    return points


def gerar_C4(n_samples: int):
    points = []
    for _ in range(n_samples):
        x = random.uniform(0, 1)
        y = random.uniform(-1 + x, 0)
        points.append([x, y, 3])

    return points


def gerar_C5(n_samples: int):
    points = []
    for _ in range(n_samples):
        x = random.uniform(0, 1)
        y = random.uniform(1 - x, sqrt(1 - x**2))
        points.append([x, y, 4])

    return points


def gerar_C6(n_samples: int):
    points = []
    for _ in range(n_samples):
        x = random.uniform(-1, 0)
        y = random.uniform(x + 1, sqrt(1 - x**2))
        points.append([x, y, 5])

    return points


def gerar_C7(n_samples: int):
    points = []
    for _ in range(n_samples):
        x = random.uniform(-1, 0)
        y = random.uniform(-sqrt(1 - x**2), -1 - x)
        points.append([x, y, 6])

    return points


def gerar_C8(n_samples: int):
    points = []
    for _ in range(n_samples):
        x = random.uniform(0, 1)
        y = random.uniform(-sqrt(1 - x**2), -1 + x)
        points.append([x, y, 7])

    return points


def generate_data(n_samples_per_class: int):
    points = (
        gerar_C1(n_samples_per_class)
        + gerar_C2(n_samples_per_class)
        + gerar_C3(n_samples_per_class)
        + gerar_C4(n_samples_per_class)
        + gerar_C5(n_samples_per_class)
        + gerar_C6(n_samples_per_class)
        + gerar_C7(n_samples_per_class)
        + gerar_C8(n_samples_per_class)
    )
    return pd.DataFrame(points, columns=["x", "y", "target"])
