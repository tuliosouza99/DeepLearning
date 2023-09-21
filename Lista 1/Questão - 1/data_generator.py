import pandas as pd
import numpy as np


def generate_data(n_samples_per_class: int, seed: int = 42):
    rng = np.random.default_rng(seed)

    base_data = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
    )

    data = []
    for _ in range(n_samples_per_class):
        data.append(
            pd.DataFrame(
                (base_data + rng.uniform(-0.1, 0.1, size=base_data.shape)),
                columns=["x1", "x2", "x3"],
            ).assign(y=lambda df_: list(range(len(df_))))
        )

    return pd.concat(data, ignore_index=True).sample(frac=1, random_state=seed)
