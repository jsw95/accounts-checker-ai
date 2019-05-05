# from sklearn.metrics import classification_report
# from sklearn.neural_network import MLPClassifier
#
# from src.data_processing import *
#
# base_img_dir = "/home/jsw/Workspace/accounts/data/English/Img/GoodImg/Bmp/"
#
# folders = [base_img_dir + i for i in os.listdir(base_img_dir) if int(i[-2:]) <= 10]
#
# # feats, labels = create_training_set(folders)
#
# # X_train, X_test, y_train, y_test = train_test_split(
#     feats, labels, test_size=0.33, random_state=42, stratify=labels)
#
# # clf = NearestCentroid()
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
#
# print(classification_report(y_test, pred))
