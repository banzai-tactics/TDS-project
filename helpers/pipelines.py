from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

from helpers.models import models


numerical_pipe = Pipeline([
    # ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
    ('scaler', StandardScaler()), #.set_output(transform='pandas')
])

categorical_pipe = Pipeline([
    # ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('numerical', numerical_pipe, make_column_selector(dtype_include=['int', 'float'])),
    ('categorical', categorical_pipe, make_column_selector(dtype_include=['object'])),
])

def get_adult_pipelines():
    pipelines = {}
    for n, model in models.items():
        pipelines[n] = Pipeline([
            ('column_transformer', preprocessor),
            ('model', model)
        ])
    return pipelines