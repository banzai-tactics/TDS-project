import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector

import matplotlib.pyplot as plt


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

def preprocess_cirrhosis(df):
    # remove redundant ID column
    df.drop(columns="ID", inplace=True)
    # define columns type
    dtypes_df = df.dtypes
    # fill missing data
    numerical_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    final_imputer = ColumnTransformer([
        ('numerical', numerical_imputer, make_column_selector(dtype_include=['int', 'float'])),
        ('categorical', categorical_imputer, make_column_selector(dtype_include=['object'])),
    ])

    df = final_imputer.fit_transform(df)
    df = pd.DataFrame(df, columns=[c.split('__')[-1] for c in final_imputer.get_feature_names_out()]).astype(dtypes_df)
    # encode target
    target = 'Status'
    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])
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


# def get_settings_options(params_dict):
#     '''return a dictionary with all settings to run'''
#     full_settings = {}
#     methods = params_dict['augment_methods']
#     for m in methods:
#         method_params = [p for p in params_dict.keys() if m in p.split('__', -1)[:-1]]
#         other_params = [p for p in params_dict.keys() if ('__' not in p) and (p!='augment_methods')]
#         all_m_params = other_params + method_params
#         all_m_params_values = [params_dict[p] for p in all_m_params]
#         m_settings = list(itertools.product(*all_m_params_values))
#         full_settings[m] = m_settings
#     return full_settings


def infalte_randomly(X_train, y_train, threshold=50):
    df_train = pd.concat([X_train, y_train], axis=1)
    classes_gap = threshold - y_train.value_counts()
    classes_to_inflate = classes_gap[classes_gap>0]
    df_inflate = pd.DataFrame()
    for c, gap in classes_to_inflate.items():
        df_c_inflate = df_train[df_train[y_train.name]==c].sample(n=gap, replace=True, random_state=42)
        df_inflate = pd.concat([df_inflate, df_c_inflate])

    X_train_inflate = pd.concat([X_train, df_inflate.drop(y_train.name, axis=1)])
    y_train_inflate = pd.concat([y_train, df_inflate[y_train.name]])

    return X_train_inflate, y_train_inflate


def get_best_methods(results):
    best_methods = pd.DataFrame()
    for s in results.columns.get_level_values(1).unique().to_list():
        models_scores = results.xs(s, axis='columns', level=1)
        max_models = models_scores.idxmax(axis='columns')
        max_models.name = s
        max_models['overall'] = models_scores.max(axis=0).idxmax(axis=0)
        best_methods = pd.concat([best_methods, max_models], axis=1)
    return best_methods


def save_results_as_latex_tables(results, task_name):
    float_format = "{:0.4f}".format
    metrics = results.columns.get_level_values(1).unique().to_list()
    for metric in metrics:
        models_scores = results.xs(metric, axis='columns', level=1)
        latex = models_scores.to_latex(float_format=float_format, index=False)
        latex = latex.replace('\\toprule', '').replace('\\midrule', '').replace('\\bottomrule', '')
        latex = latex.split('\n')
        latex.insert(1, '\\toprule')
        latex.insert(4, '\\midrule')
        latex.insert(-2, '\\bottomrule')
        latex = '\n'.join(latex)
        
        # Write the LaTeX table to a file
        with open(f'../graphs/{task_name}/latex/{task_name}_{metric}_table.tex', 'w') as f:
            f.write(latex)


def bar_plot(results, save_task_name=None, wanted_models=None, wanted_metrics=None, wanted_methods=None):
    models_dict = {'lr': 'Linear Regression', 'ridge': 'Ridge', 'lasso': 'Lasso', 'rf': 'Random Forest', 'xgb': 'XGBoost'}
    reshape_df = results.stack(level=0)
    models_to_plot = wanted_models if wanted_models else results.index
    for model in models_to_plot:
        model_df = reshape_df.xs(model)
        if wanted_methods:
            model_df = model_df.loc[wanted_methods]
        metrics = wanted_metrics if wanted_metrics else model_df.columns
        num_of_metrics = len(metrics)
        fig, axes = plt.subplots(1, num_of_metrics, figsize=(24, 6))
        if num_of_metrics==1: axes =[axes]
        for i, m in enumerate(metrics):
            axes[i].bar(model_df.index, model_df[m])
            axes[i].set_title(m, fontsize=20)
            axes[i].tick_params(axis='x', labelrotation=45, labelsize=20)
        fig.suptitle(models_dict[model], fontsize=24)
        if save_task_name:
            plt.savefig(f'../graphs/{save_task_name}/{models_dict[model]}.png')
        plt.show()


def spider_plot(results, model, wanted_variables, values_names, title, save_task_name=None):
    '''
    results: pd.DataFrame
        results dataframe

    wanted_variables: list
        method to show

    values_names: list or dict
        metrics to show. if dict rename 
    '''
    if not isinstance(values_names, dict):
        if isinstance(values_names, (list, tuple)):
            values_names = dict(zip(values_names, values_names))
        else:
            raise Exception('wrong values_names input')
        
    num_vars = len(values_names)
    # Compute angle of each axis in the plot (divide the plot / number of variable)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Ensure the plot is a full circle
    angles += angles[:1]

    # Initialise the spider plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Makes sure the First metric is at the top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], list(values_names.values()))
    ax.set_rlabel_position(0)
    plt.ylim(0.0, 1.0)
    values = []
    for v in values_names:
        models_scores = results.xs(v, axis='columns', level=1)
        df2 = models_scores[wanted_variables]
        values.append(df2.loc[model])
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=wanted_variables)
    # ax.fill(angles,values, alpha=0.1)

    # Add legend
    plt.title(title, size=20, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    # plt.show()
    if save_task_name:
        plt.savefig(f'../graphs/{save_task_name}/{title}.png')