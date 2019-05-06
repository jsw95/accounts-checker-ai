import os
import numpy as np
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from src.dataset_management import create_training_set

from src.torch_model import *

base_data_path = "/home/jack/Workspace/data/accounts/images/"

joined = create_training_set()
# joined.to_csv

feats = np.array([i for i in joined.img])
# feats = np.reshape(feats, (feats.shape[0], feats.shape[-1]))
labels = np.array(joined.label)

X_train, X_test, y_train, y_test = train_test_split(
    feats, labels, test_size=0.33, random_state=42, stratify=labels)


print("split data")


net = Net()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []


train(2, X_train, y_train)
test(X_test, y_test)

# clf = NearestCentroid()
# clf = MLPClassifier()
# clf.fit(X_train, y_train)
#
# pred = clf.predict(X_test)
#
# print("Size of feature vector", X_train.shape)
# print("Size of label", y_train.shape)
#
# print("Actual", y_test)
# print("Predict", pred)
#
# # joblib.dump(clf, '../models/MLP-digits.sav')

print(classification_report(y_test, pred))
