import numpy as np
from sortedcontainers import SortedList
import pandas as pd

class KNN(object):
    def __init__(self,k):
        self.k = k

    def fit(self,x,y):
        self.x = x 
        self.y = y

    def predict(self,X):
        y = np.zeros(len(X))
        for i,x in enumerate(X):
            sl = SortedList(load=self.k)
            for j, xt in enumerate(self.x):
                diff = x-xt
                d = diff.dot(diff)
                if len(sl) < self.k:
                    sl.add((d, self.y[j]))
                else:
                    if d < sl[-1][0]:
                        del sl[-1]
                        sl.add((d, self.y[j]))
            votes = {}
            for _,v in sl:
                votes[v] = votes.get(v,0) + 1
            max_votes = 0
            max_votes_class = -1
            for v,count in votes.items():
                if count > max_votes:
                    max_votes = count
                    max_votes_class = v
            y[i] = max_votes_class

        return y

    def score(self,X,Y):
        yhat= self.predict(X)
        return np.mean(yhat == Y)


