import numpy as np
import pandas as pd


def get_confusion_matrix_df(arr: np.ndarray, idx:list = None):
    if idx is None:
        idx = ['Feet', 'Left Hand', 'Right Hand']
    df = pd.DataFrame(arr, index=idx, columns=idx)
    return df
