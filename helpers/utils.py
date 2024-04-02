import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import get_scorer#roc_auc_score, accuracy_score, f1_score, 
from sklearn.model_selection import train_test_split, GridSearchCV


def preprocess_adult(df):
    df.replace('?', np.nan, inplace=True)

    le = LabelEncoder()
    df['income'] = le.fit_transform(df['income'])

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df.drop(['educational-num', 'capital-gain', 'capital-loss'], axis=1, inplace=True)

    return df


def evaluate_model(model, X_test, y_test, scoring):
    scorer = get_scorer(scoring)._score_func    
    try:
        score_probas = scorer(y_test, model.predict_proba(X_test)[:, 1]) 
    except:
        score_probas = 0
    try:
        score_preds = scorer(y_test, model.predict(X_test)) 
    except:
        score_preds = 0

    score = max(score_probas, score_preds)
    return score

def fit_and_evaluate(X, y, search_estimators, search_params, scoring='roc_auc', test_size_proportion=0.33, verbosity=0):
    ''' fit and evaluate several models
    
    Parameters
    ----------
    X : pd.DataFrame
        The data to fit the model to

    y : pd.Series
        The corresponding label for each sample in X

    search_estimators : dict. <str, Pipeline/Model>
        Dictionary whose keys are models name and values are a pipeline or model

    search_params : dict. <str, dict <str, list>>
        Dictionary whose keys are models name and values are dictionaries of possible parameters for the estimators

    scoring : str
        The scoring function to maximize in fit

    test_size_proportion : float
        The ratio of the test split

    verbosity : float
        Controls the verbosity. the higher, the more messages

    Returns
    -------
    best_estimators: dict
        Dictionary whose keys are models names and values are the best estimators we found
    
    best_estimators: dict
        Dictionary whose keys are models names and values are the best scores of each model
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_proportion, random_state=42)

    best_estimators = {}
    scores = {}

    for n, pipe in search_estimators.items():
        clf = GridSearchCV(pipe, search_params[n], cv=3, scoring=scoring, return_train_score=True, n_jobs=-1, verbose=verbosity)
        clf.fit(X_train, y_train)
        n_best_model = clf.best_estimator_
        score = evaluate_model(n_best_model, X_test, y_test, scoring)
        best_estimators[n] = n_best_model
        scores[n] = score

    return best_estimators, scores