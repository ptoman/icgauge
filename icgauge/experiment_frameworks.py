# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter
from scipy.stats.stats import pearsonr

import data_readers
import feature_extractors as fe
import label_transformers as lt
import training_functions as training
import utils

# May eventually add other frameworks (e.g., neural network, rule-based)


def build_dataset(reader, phi_list, class_func, vectorizer=None, verbose=False):
    """Core general function for building experimental
    hand-generated feature datasets.
    
    Parameters
    ----------
    reader : iterator
       Should follow the format of data_readers. This is the dataset 
       we'll be featurizing.
       
    phi_list : array of feature functions (default: [`manual_content_flags`])
        Any function that takes a string as input and returns a 
        bool/int/float-valued dict as output.
       
    class_func : function on the labels
        A function that modifies the labels based on the experimental 
        design. If `class_func` returns None for a label, then that 
        item is ignored.
       
    vectorizer : sklearn.feature_extraction.DictVectorizer    
       If this is None, then a new `DictVectorizer` is created and
       used to turn the list of dicts created by `phi` into a 
       feature matrix. This happens when we are training.
              
       If this is not None, then it's assumed to be a `DictVectorizer` 
       and used to transform the list of dicts. This happens in 
       assessment, when we take in new instances and need to 
       featurize them as we did in training.
       
    Returns
    -------
    dict
        A dict with keys 'X' (the feature matrix), 'y' (the list of
        labels), 'vectorizer' (the `DictVectorizer`), and 
        'raw_examples' (the example strings, for error analysis).
    """    
    labels = []
    feat_dicts = []
    raw_examples = []
    for i, (paragraph, parse, label) in enumerate(reader()):
        if i % 25 == 0:
            print "   Starting feature extraction for unit #%d " % (i+1)
        cls = class_func(label)
        #print label, cls
        if cls != None:
            labels.append(cls)
            raw_examples.append(paragraph)

            if verbose:
                print cls, ":", paragraph
            
            features = Counter()
            for phi in phi_list:
                cur_feats = phi(paragraph, parse)
                if cur_feats is None:
                    continue
                # If we won't accidentally blow away data, merge 'em.
                overlap_feature_names = features.viewkeys() & cur_feats.viewkeys()
                if verbose and len(overlap_feature_names) > 0:
                    print "Note: Overlap features are ", overlap_feature_names
                features |= cur_feats
            feat_dicts.append(features)

            if verbose:
                print features
                print 
    print "Completed all feature extraction: %d units" % (i+1)
    
    # In training, we want a new vectorizer, but in 
    # assessment, we featurize using the existing vectorizer:
    feat_matrix = None    
    if vectorizer == None:
        vectorizer = DictVectorizer(sparse=True)
        feat_matrix = vectorizer.fit_transform(feat_dicts)
    else:
        feat_matrix = vectorizer.transform(feat_dicts)
        
    return {'X': feat_matrix, 
            'y': labels, 
            'vectorizer': vectorizer, 
            'raw_examples': raw_examples}

def experiment_features(
        train_reader=data_readers.toy, 
        assess_reader=None, 
        train_size=0.7,
        phi_list=[fe.manual_content_flags], 
        class_func=lt.identity_class_func,
        train_func=training.fit_logistic_at_with_crossvalidation,
        score_func=utils.safe_weighted_f1,
        verbose=True):
    """Generic experimental framework for hand-crafted features. 
    Either assesses with a random train/test split of `train_reader` 
    or with `assess_reader` if it is given.
    
    Parameters
    ----------
    train_reader : data iterator (default: `train_reader`)
        Iterator for training data.
       
    assess_reader : iterator or None (default: None)
        If None, then the data from `train_reader` are split into 
        a random train/test split, with the the train percentage 
        determined by `train_size`. If not None, then this should 
        be an iterator for assessment data (e.g., `dev_reader`).
        
    train_size : float (default: 0.7)
        If `assess_reader` is None, then this is the percentage of
        `train_reader` devoted to training. If `assess_reader` is
        not None, then this value is ignored.
       
    phi_list : array of feature functions (default: [`manual_content_flags`])
        Any function that takes a string as input and returns a 
        bool/int/float-valued dict as output.
       
    class_func : function on the labels
        A function that modifies the labels based on the experimental 
        design. If `class_func` returns None for a label, then that 
        item is ignored.
       
    train_func : model wrapper (default: `fit_logistic_at_with_crossvalidation`)
        Any function that takes a feature matrix and a label list
        as its values and returns a fitted model with a `predict`
        function that operates on feature matrices.
    
    score_metric : function name (default: `utils.safe_weighted_f1`)
        This should be an `sklearn.metrics` scoring function. The 
        default is weighted average F1.
        
    verbose : bool (default: True)
        Whether to print out the model assessment to standard output.
       
    Prints
    -------    
    To standard output, if `verbose=True`
        Model confusion matrix and a model precision/recall/F1 report.
        
    Returns
    -------
    float
        The overall scoring metric as determined by `score_metric`.

    np.array
        The confusion matrix (rows are truth, columns are predictions)

    list of dictionaries
        A list of {truth:_ , prediction:_, example:_} dicts on the assessment data
    
    """        
    # Train dataset:
    train = build_dataset(train_reader, phi_list, class_func, vectorizer=None, verbose=verbose) 

    # Manage the assessment set-up:
    indices = np.arange(0, len(train['y']))
    X_train = train['X']
    y_train = np.array(train['y'])
    train_examples = np.array(train['raw_examples'])
    X_assess = None 
    y_assess = None
    assess_examples = None
    if assess_reader == None:
         print "   Raw y training distribution:"
         print "  ", np.bincount(y_train)[1:]

         indices_train, indices_assess, y_train, y_assess = train_test_split(
                indices, y_train, train_size=train_size, stratify=y_train)

         X_assess = X_train[indices_assess]
         assess_examples = train_examples[indices_assess]

         X_train = X_train[indices_train]
         train_examples = train_examples[indices_train]

         print "   Train y distribution:"
         print "  ", np.bincount(y_train)[1:]
         print "   Test y distribution:"
         print "  ", np.bincount(y_assess)[1:]
  
    else:
        assess = build_dataset(
            assess_reader, 
            phi_list, 
            class_func, 
            vectorizer=train['vectorizer'])
        X_assess, y_assess, assess_examples = assess['X'], assess['y'], np.array(assess['raw_examples'])

    # Train:      
    mod = train_func(X_train, y_train)    
    
    # Predictions:
    predictions_on_assess = mod.predict(X_assess)
    assess_performance = get_score_example_pairs(y_assess, predictions_on_assess, assess_examples)

    predictions_on_train = mod.predict(X_train)
    train_performance = get_score_example_pairs(y_train, predictions_on_train, train_examples)
    
    # Report:
    if verbose:
        print "\n-- TRAINING RESULTS --"
        print_verbose_overview(y_train, predictions_on_train)
        print "\n-- ASSESSMENT RESULTS --"
        print_verbose_overview(y_assess, predictions_on_assess)

    # Return the overall results on the assessment data:
    return score_func(y_assess, predictions_on_assess), confusion_matrix(y_assess, predictions_on_assess), assess_performance
    

def get_score_example_pairs(y, y_hat, examples):
    """ Return a list of dicts: {truth score, predicted score, example} """
    paired_results = sorted(zip(y, y_hat), key=lambda x: x[0]-x[1])
    performance = []
    for i, (truth, prediction) in enumerate(paired_results): 
        performance.append({"truth": truth, "prediction": prediction, "example": examples[i]})
    return performance


def print_verbose_overview(y, yhat):
    """ Print a performance overview """
    print "Correlation: ", pearsonr(y, yhat)[0]
    print "Classification report:"
    print classification_report(y, yhat, digits=3)
    print "Confusion matrix:"
    print confusion_matrix(y, yhat)
    print "  (Rows are truth; columns are predictions)"


def experiment_features_iterated(
        train_reader=data_readers.toy, 
        assess_reader=None, 
        train_size=0.7,
        phi_list=[fe.manual_content_flags], 
        class_func=lt.identity_class_func,
        train_func=training.fit_logistic_at_with_crossvalidation,
        score_func=utils.safe_weighted_f1,
        verbose=True,
        iterations=1):
    """
    Generic iterated experimental framework for hand-crafted features. 
    """  
    correlation_overall = []
    conf_matrix_overall = None
    assess_performance = []
    while len(correlation_overall) < iterations:
        print "\nStarting iteration: %d/%d" % (len(correlation_overall)+1, iterations)
        try:
            correlation_local, conf_matrix_local, perf_local = experiment_features(
                train_reader=train_reader, 
                assess_reader=assess_reader, 
                train_size=train_size,
                phi_list=phi_list, 
                class_func=class_func,
                train_func=train_func,
                score_func=score_func,
                verbose=verbose)
                
            correlation_overall.append(correlation_local[0])
            assess_performance.extend(perf_local)
            
            if conf_matrix_overall is None:
                conf_matrix_overall = conf_matrix_local
            else: 
                conf_matrix_overall += conf_matrix_local
        except (ValueError,UserWarning) as e:
            print e
    
    if verbose:
        print "\n-- OVERALL --"
        print correlation_overall
        print conf_matrix_overall
    
    return correlation_overall, conf_matrix_overall, assess_performance
