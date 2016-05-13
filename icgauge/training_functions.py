# -*- coding: utf-8 -*-
#!/usr/bin/python

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import mord

def fit_classifier_with_crossvalidation(X, y, basemod, cv, param_grid, 
                                        scoring='f1_macro', verbose=False): 
    """Fit a classifier with hyperparmaters set via cross-validation.

    Parameters
    ----------
    X : 2d np.array
        The matrix of features, one example per row.
        
    y : list
        The list of labels for rows in `X`.  
    
    basemod : an sklearn model class instance
        This is the basic model-type we'll be optimizing.
    
    cv : int
        Number of cross-validation folds.
        
    param_grid : dict
        A dict whose keys name appropriate parameters for `basemod` and 
        whose values are lists of values to try.
        
    scoring : value to optimize for (default: f1_macro)
        Other options include 'accuracy' and 'f1_micro'. See
        http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            
    Prints
    ------
    To standard output:
        The best parameters found.
        The best macro F1 score obtained.
        
    Returns
    -------
    An instance of the same class as `basemod`.
        A trained model instance, the best model found.
    """    
    # Find the best model within param_grid:
    crossvalidator = GridSearchCV(basemod, param_grid, cv=cv, scoring=scoring)
    crossvalidator.fit(X, y)
    if verbose:
        # Report some information:
        print("Best params", crossvalidator.best_params_)
        print("Best score: %0.03f" % crossvalidator.best_score_)
    # Return the best model found:
    return crossvalidator.best_estimator_
    
def fit_maxent_with_crossvalidation(X, y):
    """A classification model of dataset with hyperparameter 
    cross-validation. Maximum entropy/logistic regression variant.
    
    Some notes:
        
    * 'fit_intercept': whether to include the class bias feature.
    * 'C': weight for the regularization term (smaller is more regularized).
    * 'penalty': type of regularization -- roughly, 'l1' ecourages small 
      sparse models, and 'l2' encourages the weights to conform to a 
      gaussian prior distribution.
    
    Other arguments can be cross-validated; see 
    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    
    Parameters
    ----------
    X : 2d np.array
        The matrix of features, one example per row.
        
    y : list
        The list of labels for rows in `X`.   
    
    Returns
    -------
    sklearn.linear_model.LogisticRegression
        A trained model instance, the best model found.
    """    
    
    basemod = LogisticRegression()
    cv = 5
    param_grid = {'fit_intercept': [True, False], 
                  'C': [0.4, 0.6, 0.8, 1.0, 2.0, 3.0],
                  'penalty': ['l1','l2']}    
    return fit_classifier_with_crossvalidation(X, y, basemod, cv, param_grid,
                                               verbose=False)
    
    
def fit_logistic_it_with_crossvalidation(X, y):
    """An ordinal model of dataset with hyperparameter 
    cross-validation.  Immediate-Threshold (logistic/threshold) variant.
    
    Parameters & returns as per other training functions.
    """    
    
    basemod = mord.LogisticIT()
    cv = 5
    param_grid = {'alpha': [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]}    
    return fit_classifier_with_crossvalidation(X, y, basemod, cv, param_grid,
                                               verbose=False)
                                               
def fit_logistic_at(X, y):
    """An ordinal model of dataset without hyperparameter 
    cross-validation -- uses defaults.  
    All-Threshold (logistic/threshold) variant, recommended over 
    Intermediate-Threshold variant in Rennie and Srebro 2005.
    
    Parameters & returns as per other training functions.
    """    
    
    basemod = mord.LogisticAT() 
    basemod.fit(X,y)
    return basemod

def fit_logistic_at_with_crossvalidation(X, y):
    """An ordinal model of dataset with hyperparameter 
    cross-validation.  All-Threshold (logistic/threshold) variant.
    Recommended over Intermediate-Threshold variant in Rennie and Srebro 2005.
    
    Parameters & returns as per other training functions.
    """    
    
    basemod = mord.LogisticAT()
    cv = 5
    param_grid = {'alpha': [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]}    
    return fit_classifier_with_crossvalidation(X, y, basemod, cv, param_grid,
                                               verbose=False)
                                               
def fit_logistic_or_with_crossvalidation(X, y):
    """An ordinal model of dataset with hyperparameter 
    cross-validation.  Ordinal Ridge (regression) variant.
    
    Parameters & returns as per other training functions.
    """    
    
    basemod = mord.OrdinalRidge()
    cv = 5
    param_grid = {'fit_intercept': [True, False], 
                  'alpha': [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0],
                  'normalize': [True, False]}    
    return fit_classifier_with_crossvalidation(X, y, basemod, cv, param_grid,
                                               verbose=False)
                                               
def fit_logistic_mcl_with_crossvalidation(X, y):
    """An ordinal model of dataset with hyperparameter 
    cross-validation.  Multiclass Logistic (logistic/classification) variant.
    
    Parameters & returns as per other training functions.
    """    
    
    basemod = mord.MulticlassLogistic()
    cv = 5
    param_grid = {'alpha': [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]}
    return fit_classifier_with_crossvalidation(X, y, basemod, cv, param_grid,
                                               verbose=False)
