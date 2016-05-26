# -*- coding: utf-8 -*-
#!/usr/bin/python


from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import mord

def fit_classifier_with_crossvalidation(X, y, basemod, cv, param_grid, 
                                        scoring='r2', verbose=False):
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
        
    scoring : value to optimize for (default: accuracy)
        What we optimize for.  Best to choose "accuracy" or "r2".
        The F1 variants are meaningless for this problem since so few
        models predict in every category. "roc_auc", "average_precision",
        "log_loss" are unsupported.
            
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
    # Report some information:
    for combination in crossvalidator.grid_scores_:
        print combination
    #print("Best params", crossvalidator.best_params_)
    #print("Best score: %0.03f" % crossvalidator.best_score_)
    # Return the best model found:
    return crossvalidator.best_estimator_

def fit_maxent(X, y, C = 1.0):
    """A classification model of dataset. L2 regularized.
       C : float, optional (default=1.0)
           Inverse of regularization strength; must be a positive float. 
           Like in support vector machines, smaller values specify 
           stronger regularizatioN
    """   
    basemod = LogisticRegression(penalty='l2', C = C)
    basemod.fit(X,y)
    return basemod    

def fit_maxent_balanced(X, y, C = 1.0):
    """A classification model of dataset. L2 regularized & forces balanced classes.
       C : float, optional (default=1.0)
           Inverse of regularization strength; must be a positive float. 
           Like in support vector machines, smaller values specify 
           stronger regularizatioN
    """   
    basemod = LogisticRegression(penalty='l2', class_weight='balanced', C = C)
    basemod.fit(X,y)
    return basemod
    

def fit_maxent_with_crossvalidation(X, y, C = 1.0):
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
    
    basemod = LogisticRegression(penalty='l2', C = C)
    cv = 5
    param_grid = {'fit_intercept': [True, False], 
                  'C': [0.4, 0.6, 0.8, 1.0, 2.0, 3.0],
                  'penalty': ['l1','l2']}    
    return fit_classifier_with_crossvalidation(X, y, basemod, cv, param_grid,
                                               verbose=False)
    
    
def fit_logistic_it_with_crossvalidation(X, y, alpha = 1.0):
    """An ordinal model of dataset with hyperparameter 
    cross-validation.  Immediate-Threshold (logistic/threshold) variant.
    
    Parameters & returns as per other training functions.
    
    alpha: float :
        Regularization parameter. Zero is no regularization, 
        higher values increate the squared l2 regularization.
    """    
    
    basemod = mord.LogisticIT(alpha = alpha)
    cv = 5
    param_grid = {'alpha': [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]}    
    return fit_classifier_with_crossvalidation(X, y, basemod, cv, param_grid,
                                               verbose=False)
                                               
def fit_logistic_at(X, y, alpha = 1.0):
    """An ordinal model of dataset without hyperparameter 
    cross-validation -- uses defaults.  
    All-Threshold (logistic/threshold) variant, recommended over 
    Intermediate-Threshold variant in Rennie and Srebro 2005.
    
    Parameters & returns as per other training functions.
    alpha: float :
        Regularization parameter. Zero is no regularization, 
        higher values increate the squared l2 regularization.
    """    
    
    basemod = mord.LogisticAT(alpha = alpha)
    basemod.fit(X,y)
    return basemod

def fit_logistic_at_6(X, y):
    return fit_logistic_at(X, y, 6.0)

def fit_logistic_at_with_crossvalidation(X, y, alpha = 1.0):
    """An ordinal model of dataset with hyperparameter 
    cross-validation.  All-Threshold (logistic/threshold) variant.
    Recommended over Intermediate-Threshold variant in Rennie and Srebro 2005.
    
    Parameters & returns as per other training functions.
    alpha: float :
        Regularization parameter. Zero is no regularization, 
        higher values increate the squared l2 regularization.
    """    
    
    basemod = mord.LogisticAT(alpha = alpha)
    cv = 3
    param_grid = {'alpha': [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]}    
    return fit_classifier_with_crossvalidation(X, y, basemod, cv, param_grid,
                                               verbose=False)
                                               
def fit_logistic_or_with_crossvalidation(X, y, alpha = 1.0):
    """An ordinal model of dataset with hyperparameter 
    cross-validation.  Ordinal Ridge (regression) variant.
    
    Parameters & returns as per other training functions.
    alpha: float :
        Regularization parameter. Zero is no regularization, 
        higher values increate the squared l2 regularization.
    """    
    
    basemod = mord.OrdinalRidge(alpha = alpha)
    cv = 5
    param_grid = {'fit_intercept': [True, False], 
                  'alpha': [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0],
                  'normalize': [True, False]}    
    return fit_classifier_with_crossvalidation(X, y, basemod, cv, param_grid,
                                               verbose=False)
                                               
def fit_logistic_mcl_with_crossvalidation(X, y, alpha = 1.0):
    """An ordinal model of dataset with hyperparameter 
    cross-validation.  Multiclass Logistic (logistic/classification) variant.
    
    Parameters & returns as per other training functions.
    alpha: float :
        Regularization parameter. Zero is no regularization, 
        higher values increate the squared l2 regularization.
    """    
    
    basemod = mord.MulticlassLogistic(alpha = alpha)
    cv = 5
    param_grid = {'alpha': [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]}
    return fit_classifier_with_crossvalidation(X, y, basemod, cv, param_grid,
                                               verbose=False)
