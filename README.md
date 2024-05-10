# Data augmentation using Counterfactuals explanations
Final project in Tabular Data Science course  

# Goal
To improve performance on small and imbalanced datasets by over sampling using counterfactual methods.

# Background
Small and imbalanced datasets pose a big problem for real day to day usage. They lack the ability to  generalize well, tend to overfit, be biased and lack in performance.
One popular way of solving these issues is by augumenting the data. In this project we use counterfactuals which originated from the eXplainapiltiy field to augement the data.
 
This solution is model and data agnostic and improves performance compared to previous methods.

# Installation
The relevant packages to run the project are specified in `requirements.txt` and can be seen below in [Requirements Section.](#requirements)
For mac compatability, first do `brew install libomp`

# Getting started
Using our method is simple. There are two steps, first you need to initiate a `DataAugmentor` object, providing it all the relevant parameters, such the data to augment and which method to augment. Second, use the `augment` method to return the augmented data. As seen below.

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
We implemented an all-in-one object to augment the data with different oversampling methods. Each method can be utilized in two ways, either to balance the data or to sample randomly from it. Note that in counterfactual methods, you can't balance the regression task.
This works for any type of data - binary classification, multi-class classification and regression. The supported methods are:
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
