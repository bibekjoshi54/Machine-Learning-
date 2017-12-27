import numpy as  np
import matplotlib.pyplot as plt
# from Util import util


def get_data():
    w = np.array([-0.01,0.23])
    b = 0.1
    X = np.random.random((300, 2)) * 2 -1
    y = np.sign(X.dot(w) + b)
    return X,y

class Perceptron:
    def fit(self, X,Y, learning_rate=1.0,epochs = 1000):
        D = X.shape[1]
        self.w = np.random.randn(D)
        self.b = 0

        N = len(Y)
        cost = []

        for epoch in range(epochs):
            Yhat = self.predict(X)
            incorrect = np.nonzero(Y != Yhat)[0]
            if len(incorrect) == 0:
                break
            i = np.random.choice(incorrect)
            self.w += learning_rate*Y[i]*X[i]
            self.b += learning_rate*Y[i]

            c = len(incorrect) / float(N)
            cost.append(c)

        print("final w:",self.w , "final b:", self.b, "epochs:", (epoch +1),"/",epochs)

        plt.plot(cost)
        plt.show()


    def predict(self, X):
        return np.sign(X.dot(self.w)+ self.b)
        
    def score(self,X,Y):
        P = self.predict(X)
        return np.mean(P==Y)


if __name__=='__main__':
    X,y = get_data()
    plt.scatter(X[:,0],X[:,1],c=y,s=100,alpha=0.5)
    plt.show()

    Ntrain = int(len(y)/2)
    print(Ntrain)
    XTrain = X[:Ntrain]
    yTrain = y[:Ntrain]
    XTest,yTest = X[Ntrain:],y[Ntrain:]
    

    model = Perceptron()
    model.fit(X[:Ntrain],y[:Ntrain],epochs = 4000)

    print('Train Accuracy', model.score(XTrain,yTrain))
    print('Test Accuracy: ', model.score(XTest,yTest))