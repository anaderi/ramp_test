import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import f1_score

train_filename = '../data/public/public_train.csv'

p_threshold = 0.001


def score(y_test, y_pred):
    return f1_score(y_test > p_threshold, y_pred > p_threshold)


def train_submission(module_path, X_array, y_array, train_is):
    regressor = import_module('regressor', module_path)
    reg = regressor.Regressor()
    reg.fit(X_array[train_is], y_array[train_is])
    return reg


def test_submission(trained_model, X_array, test_is):
    reg = trained_model
    y_pred = reg.predict(X_array[test_is])
    return y_pred


if __name__ == '__main__':
    print("Reading file ...")
    data = pd.read_csv('../data/public/public_train.csv')
    features = data.drop(['p-value'], axis=1)
    X_array = features.values.astype(np.float32)
    y_array = data['p-value'].values.astype(np.float32)
    skf = ShuffleSplit(len(y_array), n_iter=2, test_size=0.5, random_state=67)
    print("Training model ...")
    for train_is, test_is in skf:
        trained_model = train_submission('.', X_array, y_array, train_is)
        y_pred = test_submission(trained_model, X_array, test_is)
        print 'f1 = ', score(y_array[test_is], y_pred)
