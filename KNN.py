# Steps to make a prediction:
# 1. Find K closest neighbors
# 1.1. Measure the distance between two points: Euclidean distance; Cosine similarity
# 2. Use the neighbors for prediction

# Implementation:
# 1. Obtaining the data
# 2. Querying the nearest neighbors
# *. KNN is a non-parametric algorithm, so training and predicting share the data.

from collections import Counter
import math

class KNN:
    def __init__(self):
        """
        # of features: n + 1
        # of samples: m + 1
        X = [[x_00, x_01, ..., x_0n], ... [x_m0, x_m1, ..., x_mn]]
        y = [y_0, y_1, ..., y_m]
        """ 
        self.X = None
        self.y = None

    def train(self, X, y):
        self.X = X # Space & Time complexity: O(1)
        self.y = y # Space & Time complexity: O(1)

    def predict(self, X_new, k, predict_type="regression"):
        distance_label = [
            (self.distance(X_new, X), y) # Time complexity: O(n) - features
            for X, y                     # Time complexity: O(m) - data points
            in zip(self.X, self.y)
        ]                                # Total Time complexity: O(m*n); Space complexity: O(m)
        kNeighbors = sorted(distance_label)[:k] # Time complexity: O(mlog(m)) Space complexity: O(log(m))
        
        if predict_type == "regression":
            # regression 
            return sum([y for _, y in kNeighbors])/k
        if predict_type == "classification":
            # classification
            return Counter([y for _, y in kNeighbors]).most_common(1)[0][0]

    def distance(self, X1, X2):
        sqrt_sum = 0
        for i in range(len(X1)):
            sqrt_sum += (X1[i]-X2[i])**2
        return math.sqrt(sqrt_sum)


X_train = [[0, 0], [0, 1], [0, 2], [1, 1], [2, 2]]
y_train = [0, 0, 1, 1, 1]
X_new = [2, 1.9]
k = 2
knn = KNN()
knn.train(X_train, y_train)
res = knn.predict(X_new, k, predict_type="regression")
print(res)

# follow-up question: How to determine optimal k value:
# 1. simplest approach k = sqrt(No. data points)
# 2. cross-validation to choose optimal value of k (hyperparameter)