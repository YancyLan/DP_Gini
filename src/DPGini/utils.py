from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def sort_X(X):
    return np.sort(np.array(X))

def cal_gini(x_sorted):
    n = len(x_sorted)
    w = 2 * np.arange(1, n + 1) - n - 1
    return n/(n-1)*(w * x_sorted).sum() / (n * x_sorted.sum())


