import itertools
import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV


def preprocess_adult(df):
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    le = LabelEncoder()
    df['income'] = le.fit_transform(df['income'])

    # oe = OrdinalEncoder()
    # str_cols = df.select_dtypes(include='object').columns.tolist()
    # df[str_cols] = oe.fit_transform(df[str_cols])

    df.drop(['educational-num', 'capital-gain', 'capital-loss'], axis=1, inplace=True)

    return df

def preprocess_german(df):
    df.fillna('NA', inplace=True)

    le = LabelEncoder()
    df['Risk'] = le.fit_transform(df['Risk'])

    return df

def evaluate_model(model, X_test, y_test, scoring):
    # scorer = get_scorer(scoring)._score_func    
    # try:
    #     score_probas = scorer(y_test, model.predict_proba(X_test)[:, 1]) 
    # except:
    #     score_probas = 0
    # try:
    #     score_preds = scorer(y_test, model.predict(X_test)) 
    # except:
    #     score_preds = 0

    # score = max(score_probas, score_preds)
    # NEW
    scorer = get_scorer(scoring)   
    score = scorer(model, X_test, y_test)
    return score


def fit_and_evaluate(X_train, y_train, X_test, y_test, search_estimators, search_params, scoring='roc_auc', verbosity=0):
    ''' fit and evaluate several models
    
    Parameters
    ----------
    X_train : pd.DataFrame
        The data to fit the model to

    y_train : pd.Series
        The corresponding label for each sample in X_train

    X_test : pd.DataFrame
        The data to test the model fit

    y_test : pd.Series
        The corresponding label for each sample in X_test

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
    
    scores: dict
        Dictionary whose keys are models names and values are the best scores of each model
    '''
    
    best_estimators = {}
    scores = {}
    for n, pipe in search_estimators.items():
        # clf = GridSearchCV(pipe, search_params.get(n, []), cv=3, scoring=scoring, return_train_score=True, n_jobs=-1, verbose=verbosity)
        # clf.fit(X_train, y_train)
        # n_best_model = clf.best_estimator_
        # score = evaluate_model(n_best_model, X_test, y_test, scoring)
        # best_estimators[n] = n_best_model
        # scores[n] = score
        # NEW
        if isinstance(scoring, str):
            scoring = [scoring]
        clf = GridSearchCV(pipe, search_params.get(n, []), cv=3, scoring=scoring, return_train_score=True, n_jobs=-1, verbose=verbosity, refit=False)
        clf.fit(X_train, y_train)

        best_indexes = {s: np.argmax(clf.cv_results_[f'mean_test_{s}']) for s in scoring}

        best_params = {s: clf.cv_results_['params'][best_indexes[s]] for s in scoring}

        n_best_models = {s: clone(pipe).set_params(**best_params[s]) for s in scoring}
        n_best_models = {s: p.fit(X_train, y_train) for s, p in n_best_models.items()}

        n_scores = {s: evaluate_model(n_best_models[s], X_test, y_test, s) for s in scoring}

        best_estimators[n] = n_best_models
        scores[n] = n_scores


    return best_estimators, scores


def get_settings_options(params_dict):
    '''return a dictionary with all settings to run'''
    full_settings = {}
    methods = params_dict['augment_methods']
    for m in methods:
        method_params = [p for p in params_dict.keys() if m in p.split('__', -1)[:-1]]
        other_params = [p for p in params_dict.keys() if ('__' not in p) and (p!='augment_methods')]
        all_m_params = other_params + method_params
        all_m_params_values = [params_dict[p] for p in all_m_params]
        m_settings = list(itertools.product(*all_m_params_values))
        full_settings[m] = m_settings
    return full_settings