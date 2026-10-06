from pathlib import Path

import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "Table_for_task1.csv"

def get_top_similar_pairs(matrix, top_n=5):
    pairs = matrix.unstack()

    pairs = pairs[pairs.index.get_level_values(0) != pairs.index.get_level_values(1)]
    unique_pairs = pairs.groupby(
        lambda x: tuple(sorted(x)), group_keys=False
    ).first()

    top_pairs = unique_pairs.sort_values(ascending=False).head(top_n)

    return top_pairs

df = pd.read_csv(CSV_FILE, sep=";", index_col=0)

matrix_rows = df.values
rows_lengths = np.linalg.norm(matrix_rows, axis=1, keepdims=True)
raw_rows = np.divide(
    matrix_rows,
    rows_lengths,
    where=rows_lengths != 0
)
df_norm_rows = pd.DataFrame(raw_rows, index=df.index, columns=df.columns)

Rows_cos = df_norm_rows @ df_norm_rows.T

columns_lengths = np.linalg.norm(matrix_rows, axis=0, keepdims=True)
raw_columns = np.divide(
    matrix_rows,
    columns_lengths,
    where=columns_lengths != 0
)

df_norm_columns = pd.DataFrame(raw_columns, index=df.index, columns=df.columns)

Columns_cos = df_norm_columns.T @ df_norm_columns

print("Топ 2 пары самых похожих продуктов")
print(get_top_similar_pairs(Rows_cos, top_n=2))

print("Топ 2 пары самых похожих пользователей")
print(get_top_similar_pairs(Columns_cos, top_n=2))