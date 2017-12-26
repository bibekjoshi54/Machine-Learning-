import pandas as pd
import numpy as np
def get_data(name,scaleFactor = None,limit=None):
    print("Reading in and transforming data...")
    df = pd.read_csv(name)
    data = df.as_matrix()
    np.random.shuffle(data)
    if scaleFactor is not None:
        X = data[:, 1:] / scaleFactor
    else:
        X = data[:, 1:]
    Y = data[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y