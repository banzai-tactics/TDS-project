
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


models = {
    'lg': LogisticRegression(random_state=42),
    'rf': RandomForestClassifier(random_state=42),
    'xgb': XGBClassifier(random_state=42)
}

parameters = {
    'lg': {
        'model__C': [0.1, 0.5, 1]
    },
    'rf': {
        'model__max_depth': [5, 7, 9, 11],
        'model__min_samples_leaf': [30, 50],
    },
    'xgb': {
        'model__max_depth': [5, 7, 9, 11], 
        'model__min_child_weight': [30, 50],
    }
}