import numpy  as np
import pandas  as pd
import perceptron

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

if __name__== '__main__':
    X,Y = get_data('Data/train.csv')
    idx = np.logical_or(Y==1 ,Y==0)
    X = X[idx]
    y = Y[idx]
    y[y==0] = -1
    Ntrain = int(len(y)/2)
    print(Ntrain)
    XTrain = X[:Ntrain]
    yTrain = y[:Ntrain]
    XTest,yTest = X[Ntrain:],y[Ntrain:]
    model = perceptron.Perceptron()
    model.fit(X[:Ntrain],y[:Ntrain],learning_rate=0.001,epochs = 40000)

    print('Train Accuracy', model.score(XTrain,yTrain))
    print('Test Accuracy: ', model.score(XTest,yTest))