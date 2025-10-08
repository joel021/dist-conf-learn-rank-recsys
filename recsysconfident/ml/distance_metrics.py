import numpy as np
import numpy.typing as t

def mae(y_true: t.NDArray[np.float64], y_pred: t.NDArray[np.float64]) -> float:
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true: t.NDArray[np.float64], y_pred: t.NDArray[np.float64]) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
