from sklearn.externals import joblib
from src.data_processing import *

def load_model(model_name):

    filename = f'../models/{model_name}.sav'
    model = joblib.load(filename)

    return model


m = load_model('MLP-digits')




base_img_dir = "/home/jsw/Workspace/accounts/images3"

folders = [base_img_dir + i for i in os.listdir(base_img_dir) if int(i[-2:]) <= 10]


feats, labels = create_training_set(folders)

X_train, X_test, y_train, y_test = train_test_split(
    feats, labels, test_size=0.33, random_state=42, stratify=labels)
