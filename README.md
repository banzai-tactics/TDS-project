# Data augmentation using counterfactuals
Final project in Tabular Data Science course

# Goal
To improve performance on small and imbalanced dataset by over sampling using counterfactuals explanations method

# Getting started
Using our method is very easy. There are two simple steps, first you need to initiate a `DataAugmentor` object, providing it all the relevant parameters, such the data to augment and which method to augment. Second, use the `augment` method to return the augmented data.

## Example usage
```
# split data etc
augmentor = DataAugmentor(X_train, y_train, X_test, y_test,
                         method='cf_random', regression=False,
                         continuous_feats=[feat1, feat2, feat3],
                         )
X_train_augmented_balanced, y_train_augmented_balanced = augmentor.augment(balance=True)
```

# Supported methods for augmenting data
We implemented all-in-one Object to augment the data with different oversampling methods. Each method can be utilized in to ways, either to balance the data or to sample randomly from it. Note that in counterfactual methods, you cant balance regression task.  
This works for any type of data - binary classification, multi-class classification and regression. the supported methods are:
* Random over sampling
* SMOTE
* counterfactuals method: three Model-agnostic methods:
* * Randomized sampling
  * KD-Tree (for counterfactuals within the training data)
  * Genetic algorithm

## Requirements
Can be seen in `requirements.txt`:
* dice_ml==0.11
* imbalanced_learn==0.12.2
* Jinja2==3.1.3
* lightgbm==4.3.0
* matplotlib==3.5.2
* numpy==1.21.5
* pandas==1.4.4
* resreg==0.2
* scikit_learn==1.4.1.post1
* xgboost==2.0.3



# Error handling - on Mac with M1/2 
when installing lightgbm first do brew install libomp
