import numpy as np
def Softmax(x : np.ndarray) -> np.ndarray :
    out = np.zeros(x.shape)
    for i in range(x.shape[0]) :
        reducedrow = x[i] - np.max(x[i])
        exprow = np.exp(reducedrow)
        sum_exprow = np.sum(exprow)
        out[i] = exprow / sum_exprow
    return out

def CrossEntropy(S : np.ndarray, y : np.ndarray) -> float :
    batch_size = S.shape[0]
    y_eyed = np.eye(S.shape[1])[y]
    return -np.sum(np.log(np.sum(S*y_eyed+1e-7, axis = 1))) / batch_size
