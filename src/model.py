from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.data_processing import *

X_train, X_test, y_train, y_test = create_training_set()

# create a model
# clf = NearestCentroid()
clf = MLPClassifier()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)


print("Size of feature vector", X_train.shape)
print("Size of label", y_train.shape)

print("Actual", y_test)
print("Predict", pred)

print(classification_report(y_test, pred))



