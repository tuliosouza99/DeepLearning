import math

import pandas as pd


def generate_data(n_samples: int, train_k: int, pred_k: int):
    data = []
    for i in range(n_samples):
        train_list = [
            math.sin(i + k)**2 + math.cos(i + k + math.cos(i + k))
            for k in range(train_k)
        ]
        pred_list = [
            pow(math.sin(i + k), 2) + math.cos(i + k + math.cos(i + k))
            for k in range(train_k, train_k + pred_k)
        ]
        data.append(train_list + pred_list)

    columns = [f'x{i}' for i in range(1, train_k + 1)] + [
        f'y{i}' for i in range(1, pred_k + 1)
    ]
    return pd.DataFrame(data, columns=columns)
