import os
import concurrent.futures
import logging

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp

# data directory
DATA_DIR = os.path.join('data',)

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])

    # subtracy mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()

    return df


def pre_process_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))

    df = standardize(df)
    print("After standardization {}".format(df.shape))

    # create dummy variables for categoricals
    df = pd.get_dummies(df)
    print("After converting categoricals:\t{}".format(df.shape))

    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})

    df.fillna(0, inplace=True)

    return df


def make_country_sub(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']

    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds[:, 1],  # proba p=1
                               columns=['poor'],
                               index=test_feat.index)

    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]


def train_and_predict(X_train, y_train, x_test, country):
    logger.debug("Training classifier for country {}".format(country))

    def objective_func(C):
        classifier = SVC(C=C, probability=True)
        scores = cross_val_score(classifier, X_train, y_train, cv=5)
        return scores.mean()

    best_parameters = fmin(objective_func, space=hp.uniform(
        "C", 1, 100), algo=tpe.suggest, max_evals=100)

    # Reconstruct classifier
    final_classifier = SVC(C=best_parameters["C"], probability=True)

    # Train classifier
    final_classifier.fit(X_train, y_train)
    predictions = final_classifier.predict_proba(x_test)

    # convert preds to data frames
    a_sub = make_country_sub(predictions, x_test, country)
    a_sub.to_csv('submission_{}.csv'.format(country))


def main():
    data_paths = {
        'A': {
            'train': os.path.join(DATA_DIR, 'A_hhold_train.csv'),
            'test':  os.path.join(DATA_DIR, 'A_hhold_test.csv')},
        'B': {
            'train': os.path.join(DATA_DIR, 'B_hhold_train.csv'),
            'test':  os.path.join(DATA_DIR, 'B_hhold_test.csv')},
        'C': {
            'train': os.path.join(DATA_DIR, 'C_hhold_train.csv'),
            'test':  os.path.join(DATA_DIR, 'C_hhold_test.csv')
        }
    }

    # load training data
    a_train = pd.read_csv(data_paths['A']['train'], index_col='id')
    b_train = pd.read_csv(data_paths['B']['train'], index_col='id')
    c_train = pd.read_csv(data_paths['C']['train'], index_col='id')

    print("Country A")
    aX_train = pre_process_data(a_train.drop('poor', axis=1))
    ay_train = np.ravel(a_train.poor)

    print("\nCountry B")
    bX_train = pre_process_data(b_train.drop('poor', axis=1))
    by_train = np.ravel(b_train.poor)

    print("\nCountry C")
    cX_train = pre_process_data(c_train.drop('poor', axis=1))
    cy_train = np.ravel(c_train.poor)

    # load test data
    a_test = pd.read_csv(data_paths['A']['test'], index_col='id')
    b_test = pd.read_csv(data_paths['B']['test'], index_col='id')
    c_test = pd.read_csv(data_paths['C']['test'], index_col='id')

    # process the test data
    a_test = pre_process_data(a_test, enforce_cols=aX_train.columns)
    b_test = pre_process_data(b_test, enforce_cols=bX_train.columns)
    c_test = pre_process_data(c_test, enforce_cols=cX_train.columns)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.submit(train_and_predict, aX_train, ay_train, a_test, "A")
        executor.submit(train_and_predict, bX_train, by_train, b_test, "B")
        executor.submit(train_and_predict, cX_train, cy_train, c_test, "C")


if __name__ == "__main__":
    main()
