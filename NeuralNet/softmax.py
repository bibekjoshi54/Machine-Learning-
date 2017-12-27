import numpy as np

def softmax(a):
    a = np.exp(a)
    try:
        return a/a.sum(axis=1,keepdims=True)
    except np.AxisError:
        return a/a.sum()
