import math
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from helpers import utils, pipelines#, models

import dice_ml

import random

import resreg

from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, SMOTEN


class DataAugmentor(object):
    '''Initialize DataAugmentor
    
    Parameters
    ----------
    X_train : pd.DataFrame
        The data to augment

    y_train : pd.Series
        The corresponding label for each sample in X_train

    X_test : pd.DataFrame
        The test data. used only for train a model to 'cf' methods

    y_test : pd.Series
        The corresponding label for each sample in X_test

    continuous_feats : list, default=None
        List of the continuous features in the data
        
    method : ['cf_random', 'cf_genetic', 'cf_kdtree', 'random', 'smote'], default 'cf_random'
        The method to use when augmenting data.
        cf_random - counter factuals method of random sampling of features
        cf_genetic - counter factuals method using a genetic algorithm
        cf_kdtree - counter factuals from the training data
        random - random over sampling of existing instances
        smote - generates synthetic samples by interpolating between existing instances.

    regression : Bool, default False
        If true, its a regression task. else, by default, its a classification task

    cf_scoring : string, default=[f1/rmse] (classification or regression respectively)
        The score metric to train the cf model
    
    kw_args : dict, default=None
        Dictionary of additional keyword arguments to pass to func.
        '''
    def __init__(self, X_train, y_train, X_test, y_test, continuous_feats=[], method='cf_random', regression=False, cf_scoring=None, kw_args={}):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.method = method
        self.regression = regression
        self.continuous_feats = continuous_feats
        self.cf_scoring = cf_scoring
        self.kw_args = kw_args

        self.df_train = pd.concat([X_train, y_train], axis=1)
        self.categorical_feats = [c for c in X_train.columns if c not in self.continuous_feats]

                 
    def augment(self, balance=True, size=0.5):
        '''Augment the data with new samples.
    
        Parameters
        ----------
        balance : Bool, default True
            Whether to balance the target.
        
        size : int or float
            number or proportion of extra samples to generate. if balance is True, ignored.
            If int, number of samples to generate.
            If float, fraction of the X_train size samples to generate.'''
        
        if self.method[:2]=='cf':
            augmented_data = self._generate_cf_samples(balance, size)
        
        elif self.method=='random':
            augmented_data = self._generate_random_samples(balance, size)
        
        elif self.method=='smote':
            augmented_data = self._generate_smote_samples(balance, size)
            
        else:
            raise Exception("method must be one of ['cf_random', 'cf_genetic', 'cf_kdtree', 'random', 'smote']")
        
        return augmented_data
    
    def _generate_cf_samples(self, balance, size):
        self._validate_cf_input()
        # train model for cf
        cf_model = self._model_for_cf()

        if balance: # whether to sample just from the minorities classes or from all classes.
            if self.regression:
                # implementing in the future. for now raise an error
                raise Exception('cant balanced a regression task')
            else: # classification
                augmented_data = pd.DataFrame()
                classes_gaps = self.y_train.value_counts().max() - self.y_train.value_counts()
                for c, gap_size in classes_gaps.items():
                    if gap_size>0:
                        X_to_sample_from = self.X_train[self.y_train != c]
                        c_augmented = self._create_cf(X_to_sample_from, gap_size, cf_model, c)
                        augmented_data = pd.concat([augmented_data, c_augmented])
        else:
            num_of_samples = self._determine_sample_size(size)
            if self.regression: # There is different here?
                augmented_data = self._create_cf(self.X_train, num_of_samples, cf_model)
            else:                    
                augmented_data = self._create_cf(self.X_train, num_of_samples, cf_model)
            
        X_train_augmented = pd.concat([self.X_train, augmented_data.drop(self.y_train.name, axis=1)]).astype(self.X_train.dtypes)
        y_train_augmented = pd.concat([self.y_train, augmented_data[self.y_train.name]])
        return X_train_augmented, y_train_augmented


    def _generate_random_samples(self, balance, size):
        if balance:
            if self.regression:
                relevance = self._relevance_values_to_over_sample_regression(balanced=balance)
                X_train_augmented, y_train_augmented = resreg.random_oversample(self.X_train, self.y_train,
                                                                                relevance, relevance_threshold=0.5,
                                                                                over='balance', random_state=42)
                X_train_augmented = pd.DataFrame(X_train_augmented, columns=self.X_train.columns).astype(self.X_train.dtypes)
                y_train_augmented = pd.Series(y_train_augmented, name=self.y_train.name)
            else: # classification
                sampler = RandomOverSampler(random_state=42) # resample all classes but the majority class
                X_train_augmented, y_train_augmented = sampler.fit_resample(self.X_train, self.y_train)
        else:
            num_of_samples = self._determine_sample_size(size)
            # if self.regression: # there isnt a difference here
            augmented_data = self.df_train.sample(n=num_of_samples, random_state=42)
            X_train_augmented = pd.concat([self.X_train, augmented_data.drop(self.y_train.name, axis=1)]).astype(self.X_train.dtypes)
            y_train_augmented = pd.concat([self.y_train, augmented_data[self.y_train.name]])

        return X_train_augmented, y_train_augmented

    def _generate_smote_samples(self, balance, size):
        if balance:
            if self.regression:
                relevance = self._relevance_values_to_over_sample_regression(balanced=balance)
                X_train_augmented, y_train_augmented = resreg.smoter(self.X_train, self.y_train,
                                                                     relevance, relevance_threshold=0.5,
                                                                     nominal=self.categorical_feats,
                                                                     over='balance', random_state=42)
                X_train_augmented = pd.DataFrame(X_train_augmented, columns=self.X_train.columns).astype(self.X_train.dtypes)
                y_train_augmented = pd.Series(y_train_augmented, name=self.y_train.name)
            else:
                if not self.continuous_feats: # SMOTENC not designed to work with only categorical features
                    sampler = SMOTEN(random_state=42)
                elif not self.categorical_feats: # SMOTENC not designed to work with only numerical features
                    sampler = SMOTE(random_state=42)
                else: # if its mix features
                    sampler = SMOTENC(random_state=42, categorical_features=self.categorical_feats) # resample all classes but the majority class
                X_train_augmented, y_train_augmented = sampler.fit_resample(self.X_train, self.y_train)

        else:
            num_of_samples = self._determine_sample_size(size)
            frac_of_samples = self._determine_sample_frac(size)
            if self.regression:
                relevance = self._relevance_values_to_over_sample_regression(balanced=balance)
                X_train_augmented, y_train_augmented = resreg.smoter(self.X_train, self.y_train,
                                                                     relevance, relevance_threshold=0.5,
                                                                     nominal=self.categorical_feats,
                                                                     over=frac_of_samples, under=0.01, 
                                                                     random_state=42)
                X_train_augmented = pd.DataFrame(X_train_augmented, columns=self.X_train.columns).astype(self.X_train.dtypes)
                y_train_augmented = pd.Series(y_train_augmented, name=self.y_train.name)
            else: # classification
                additional_samples = ((self.y_train.value_counts(normalize=True))*(num_of_samples)).astype(int)
                sample_strategy = (additional_samples + self.y_train.value_counts()).to_dict()
                if not self.continuous_feats: # SMOTENC not designed to work with only categorical features
                    sampler = SMOTEN(random_state=42)
                elif not self.categorical_feats: # SMOTENC not designed to work with only numerical features
                    sampler = SMOTE(random_state=42)
                else: # if its mix features
                    sampler = SMOTENC(random_state=42, categorical_features=self.categorical_feats)
                X_train_augmented, y_train_augmented = sampler.fit_resample(self.X_train, self.y_train)

        return X_train_augmented, y_train_augmented


    def _validate_cf_input(self):
        if self.continuous_feats is None:
            raise Exception("'continuous_feats' must be specified")
        
        if 'total_CFs' not in self.kw_args: # if didnt specify num of cf to generate
            self.kw_args['total_CFs'] = 1 # by default generate one cf per record

        if (self.method=='cf_random') and ('random_seed' not in self.kw_args):
            self.kw_args['random_seed'] = 42


    def _model_for_cf(self):
        '''defining model to generate counterfactuals.
        
        TODO: now its knn (both classification and regression). in the future add support for others
        '''
        if self.regression:
            # cf_model_pipeline = {'cf': Pipeline([('column_transformer', pipelines.preprocessor),
            #                                      ('model', KNeighborsRegressor())])}
            # cf_model_params = {'cf': {'model__n_neighbors': list(range(1,21)), 
            #                         'model__weights': ['uniform', 'distance'],
            #                         'model__p': [1,2],
            #                         'model__algorithm': ['ball_tree', 'kd_tree', 'brute'],}}
            #
            ###########
            from lightgbm import LGBMRegressor
            cf_model_pipeline = {'cf': Pipeline([('column_transformer', pipelines.preprocessor),
                                                ('model', LGBMRegressor(random_state=42, verbose=-1))])}
            cf_model_params = {'cf': {'model__max_depth': [5, 6, 7],
                                  'model__min_child_samples': [30, 50], 
                                  'model__num_leaves': [25, 55], 
                                  'model__learning_rate': [0.1, 0.3, 0.5],
                                  'model__reg_lambda': [0, 0.5, 1.5, 3],}}
            ###########
            #
            if self.cf_scoring is None: # default score is rmse
                self.cf_scoring = 'neg_root_mean_squared_error'
        else: # if classification
            # cf_model_pipeline = {'cf': Pipeline([('column_transformer', pipelines.preprocessor),
            #                                     ('model', KNeighborsClassifier())])}
            ###########
            from lightgbm import LGBMClassifier
            cf_model_pipeline = {'cf': Pipeline([('column_transformer', pipelines.preprocessor),
                                                ('model', LGBMClassifier(random_state=42, verbose=-1))])}
            cf_model_params = {'cf': {'model__max_depth': [5, 6, 7],
                                  'model__min_child_samples': [30, 50], 
                                  'model__num_leaves': [25, 55], 
                                  'model__learning_rate': [0.1, 0.3, 0.5],
                                  'model__reg_lambda': [0, 0.5, 1.5, 3],}}
            ###########
            #
            if self.cf_scoring is None: # default score is f1
                self.cf_scoring = 'f1'
        # cf_model_params = {'cf': {'model__n_neighbors': list(range(1,21)), 
        #                             'model__weights': ['uniform', 'distance'],
        #                             'model__p': [1,2],
        #                             'model__algorithm': ['ball_tree', 'kd_tree', 'brute'],}}
        #
        ###########
        best_cf_estimator, sampled_scores = utils.fit_and_evaluate(self.X_train, self.y_train, 
                                                                   self.X_test, self.y_test,
                                                                   search_estimators=cf_model_pipeline,
                                                                   search_params=cf_model_params,
                                                                   scoring=self.cf_scoring)
        print(f'model for cf {self.cf_scoring} score: {sampled_scores}')
        return best_cf_estimator['cf'][self.cf_scoring]


    def _create_cf(self, X_pool, size, cf_model, desired_y=None, print_progress=50):
        '''
        creating counterfactuals.

        Parameters
        ----------
        X_pool : pd.DataFrame
            The pool to generate counterfactuals from

        size : int
            Number of counterfactuals to generate

        cf_model : sklearn estimator
            A fitted model to use in cf generation

        desired_y : list or string
            In classification, the desired class.
            In regression, the desired range.
        '''

        # defining counterfactuals objects
        cf_method=self.method.split('_')[1]
        m_type = 'regressor' if self.regression else 'classifier'
        d = dice_ml.Data(dataframe=self.df_train, continuous_features=self.continuous_feats, outcome_name=self.y_train.name)
        m = dice_ml.Model(model=cf_model, backend="sklearn", model_type=m_type)
        exp = dice_ml.Dice(d, m, method=cf_method)

        if desired_y is not None:
            if self.regression:
                self.kw_args['desired_range'] = desired_y
            else: # classification
                self.kw_args['desired_class'] = desired_y
            
        cf_augmented_data = pd.DataFrame()
        wanted_range = None
        cf_counter = 0
        closed_list = []
        i = 0
        while cf_counter < size:
            if (cf_counter%print_progress) == 0: # print progression
                print(f'{cf_counter}/{size}') 
            if i%len(X_pool) in closed_list: # if we saw this record and we didnt succeeded to generate cf
                i+=1 # iterate to next one
                continue # skip to next one
            index = i % len(X_pool)
            try:
                if desired_y is None: # if not specify range, define default values based on sample
                    if (self.regression): # if regression 
                        self.kw_args['desired_range'] = self._get_desired_regression_range(self.y_train[X_pool.index], index)
                    elif len(self.y_train.unique())>2: # if multiclass
                        self.kw_args['desired_class'] = self._get_desired_multi_class(self.y_train[X_pool.index], index)

                e1 = exp.generate_counterfactuals(X_pool.iloc[[index]], **self.kw_args)
                cf_df = e1.cf_examples_list[0].final_cfs_df # result cf
                # if cf_df[target].iloc[0]!=minority_class: continue
                cf_augmented_data = pd.concat([cf_augmented_data, cf_df])
                cf_counter += len(cf_df)
            except:
                closed_list += [i]

            i+=1

        return cf_augmented_data
    
    def _get_desired_regression_range(self, y, idx):
        '''for regression task define desired range for a sample.
        
        we want the desired range of the counterfactuals to be
        at least one std away to the direction of the mean.'''
        y_train_min = y.min()
        y_train_max = y.max()
        y_train_mean = y.mean()
        y_train_std = y.std()
        if y.iloc[idx] >= y_train_mean:
            upper_flip = min(y.iloc[idx]-y_train_std, y_train_mean)
            lower_flip = max(y_train_min, upper_flip-y_train_std)
            wanted_range = [lower_flip, upper_flip]
        else:
            lower_flip = max(y.iloc[idx]+y_train_std, y_train_mean)
            upper_flip = min(y_train_max, lower_flip+y_train_std)
            wanted_range = [lower_flip, upper_flip]
        return wanted_range
    

    def _get_desired_multi_class(self, y, idx):
        '''for multiclass task define desired class for a sample.
        
        we want the desired class of the counterfactuals to be
        random sampled from all classes except the query one.
        
        TODO: implement.'''

        query_class = y.iloc[idx]
        classes_counts = y.value_counts()
        relevant_class = classes_counts[classes_counts.index!=query_class]
        different_class = random.sample(relevant_class.index.to_list(), 1, counts=relevant_class.values.tolist())[0]
        return different_class
    

    def _determine_sample_size(self, size):
        '''Determine number of samples to augment
        
        if size is float, size is the fraction samples generated.
        if size is int, size is the number of samples generated'''

        assert size >=0 , "size must be non-negative"
        if isinstance(size, float):
            num_of_samples = int(size*len(self.X_train))
        else:
            num_of_samples = size
        return num_of_samples
    

    def _determine_sample_frac(self, size):
        '''Determine fraction of samples to augment
        
        if size is float, size is the fraction of samples generated.
        if size is int, size is the number of samples generated'''

        assert size >=0 , "size must be non-negative"
        if isinstance(size, float):
            frac_of_samples = size
        else:
            frac_of_samples = math.ceil(size/len(self.X_train))
        return frac_of_samples
    

    def _relevance_values_to_over_sample_regression(self, balanced):
        '''In balancing regression we try to make the bell normal.
        if not balancing, sample randomly'''
        if balanced:
            y_train_mean = self.y_train.mean()
            if y_train_mean > self.y_train.median():
                relevance = [1 if y>y_train_mean else 0 for y in self.y_train]
            else:
                relevance = [1 if y<y_train_mean else 0 for y in self.y_train]
        else:
            size = len(self.y_train)
            num_ones = int(size * 0.49) # ones need to be less than zeroes
            num_zeroes = size - num_ones
            relevance = [1] * num_ones + [0] * num_zeroes
            random.shuffle(relevance)

        return relevance
