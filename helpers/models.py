
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


models = {
    'lg': LogisticRegression(random_state=42),
    'rf': RandomForestClassifier(random_state=42),
    'xgb': XGBClassifier(random_state=42)
}

parameters = {
    'lg': {
        'model__C': [0.05, 0.1, 0.5, 1, 1.5]
    },
    'rf': {
        'model__max_depth': [5, 6, 7, 9],
        'model__min_samples_leaf': [30, 50],
    },
    'xgb': {
        'model__max_depth': [5, 6, 7, 9], 
        'model__min_child_weight': [30, 50],
        'model__reg_lambda': [0.5, 1, 1.5, 3],
        # 'model__learning_rate': [0.1, 0.3, 0.5],  
    }
}


regression_models = {
    'lr': LinearRegression(),
    'ridge': Ridge(random_state=42),
    'lasso': Lasso(random_state=42),
    'rf': RandomForestRegressor(random_state=42),
    'xgb': XGBRegressor(random_state=42)
}

regression_parameters = {
    'lr': {
        'model__fit_intercept': [True]
    },
    'ridge': {
        'model__alpha': [0.05, 0.1, 0.5, 1, 1.5]#[0.5, 1, 1.5]
    },
    'lasso': {
        'model__alpha': [0.05, 0.1, 0.5, 1, 1.5]#[0.5, 1, 1.5]
    },
    'rf': {
        'model__max_depth': [5, 7, 9],
        'model__min_samples_leaf': [30, 50]
    },
    'xgb': {
        'model__max_depth': [5, 7, 9], 
        'model__min_child_weight': [30, 50],
        'model__reg_lambda': [0.5, 1, 1.5, 3],
        'model__gamma': [0, 0.5, 1],
        'model__learning_rate': [0.05, 0.1, 0.3, 0.5],  
    }
}