import pandas as pd
import numpy as np


def generate_data_a():
    return pd.DataFrame(
        [[0, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0]], columns=['x1', 'x2', 'y']
    )


def generate_data_b(n_samples: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)

    X = rng.uniform(1, 10, n_samples)
    y = np.log10(X)

    return pd.DataFrame({'x': X, 'y': y})


def generate_data_c(n_samples: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = np.linspace(1, 10, 1000)
    y = (
        (10 * (X**5))
        + (5 * (X**4))
        + (2 * (X**3))
        - (0.5 * (X**2))
        + (3 * X)
        + 2
    )

    return pd.DataFrame({'x': X, 'y': y})
