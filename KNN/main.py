import sys
from knn import KNN
mainpackage = sys.path[0].split('/')
mainpackage.pop()
sys.path.append('/'.join(mainpackage))

from Util import util
if __name__ == '__main__':
    X,y = util.get_data('Data/train.csv',2000)
    NTrain = 1000
    XTrain,yTrain = X[:NTrain],y[:NTrain]
    Xteat, yTest = X[NTrain:], y[NTrain:]
    for k in (1,2,3,4,5):
        knn = KNN(k)
        knn.fit(XTrain,yTrain)
        print("Train Accuracvy : " + str(knn.score(XTrain,yTrain)))