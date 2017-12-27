import numpy as np
import softmax
# For single hidden layer

def forward(X,W1,b1,W2,b2):
    # Hidden layer will use the sigmoid
    Z = 1/(1 + np.exp(-X.dot(W1) - b1))
    A = Z.dot(w2) + b2
    Y = softmax.softmax(A)
    return Y


def classification_rate(Y,P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct/n_total)

def predict(X,W1,b1,W2,b2):
    YGivenX = forward(X,W1,b1,W2,b2)
    return np.argmax(YGivenX,axis=1)



if __name__ == '__main__':
    NSample = 1000
    X1 = np.random.randn(NSample,2) + np.array([0,-2])
    X2 = np.random.randn(NSample,2) + np.array([2,-2])
    X3 = np.random.randn(NSample,2) + np.array([-2,2])
    Y = np.array([0]*NSample + [1]*NSample + [2]*NSample)
    X = np.vstack([X1,X2,X3])
    import matplotlib.pyplot as plt
    plt.scatter(X[:,0],X[:,1],c=Y, s=100,alpha=0.5)
    plt.show()
    D = 2
    M = 3
    K = 3
    w1 = np.random.randn(D,M)
    b1 = np.random.randn(M)
    w2 = np.random.randn(M,K)
    b2 = np.random.randn(K)
    yhat = predict(X,w1,b1,w2,b2)
    print('Classification rate is ', classification_rate(Y,yhat))
