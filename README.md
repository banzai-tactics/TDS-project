# Data augmentation using counterfactuals
Final project in Tabular Data Science course  

# Goal
To improve performance on small and imbalanced datasets by over sampling using counterfactuals explanations methods.

# Background
Small and imbalanced datasets are a big problem. They are suffer from similar challenges such that they may not generalize well, tend to overfit, biased and lacked performance.  
Sometimes augment the dataset can remedy some of the challenges.  
Our implementation of all-in-one DataAugmentor, including our novel implemetation using counterfactuals examples can help to tackle those challenges.  
The solution is model and data agnostic (works for all complex models and all kind of datasets).

# Installations
The relevant packages to run the project are specified in `requirements.txt` and can be seen below in [Requirements Section.](#requirements)
For mac compatability, first do `brew install libomp`

# Getting started
Using our method is very easy. There are two simple steps, first you need to initiate a `DataAugmentor` object, providing it all the relevant parameters, such the data to augment and which method to augment. Second, use the `augment` method to return the augmented data.

## Example usage
```python
# ...split data and etc.
augmentor = DataAugmentor(X_train, y_train, X_test, y_test,
                         method='cf_random', regression=False,
                         continuous_feats=[feat1, feat2, feat3],
                         )
X_train_augmented_balanced, y_train_augmented_balanced = augmentor.augment(balance=True)
```
For more details, check out the [introduction notebook](data_augmentation_intro.ipynb).  
For further information, here are some deeper experiments, on different datasets:
* [Adult Income](experiments/classification_adult.ipynb) and [German Credit](experiments/experiment(german).ipynb) Datasets - augmenting for binary classification tasks.
* [Cirrhosis Prediction](experiments/multi-cirrhosis.ipynb) and [Synthetic](experiments/multi-artificial.ipynb) Dataset - augmenting for multi-class classification tasks.
* [Diabetes](experiments/regression.ipynb) Dataset - augmenting for regression tasks.

# Supported methods for augmenting data
We implemented all-in-one Object to augment the data with different oversampling methods. Each method can be utilized in two ways, either to balance the data or to sample randomly from it. Note that in counterfactual methods, you can't balance regression task.  
This works for any type of data - binary classification, multi-class classification and regression. the supported methods are:
* Random over sampling
* SMOTE
* counterfactuals method (three Model-agnostic methods):
  * Randomized sampling
  * KD-Tree (for counterfactuals within the training data)
  * Genetic algorithm


# Requirements
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
