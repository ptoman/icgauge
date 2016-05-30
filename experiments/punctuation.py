# -*- coding: utf-8 -*-
#!/usr/bin/python

#################################################
# Usage:
# From top level (/path/to/repo/icgauge), issue the command 
#      `python -m experiments.toy`
#
# Toy experiment -- machinery is hooked up and 
# working. Not a baseline experiment because it 
# doesn't use a real train vs. dev set. Insufficient
# data to get good metrics.
#################################################

import cProfile
import json

import scipy
import numpy as np
import icgauge
from icgauge import experiment_frameworks


print "Punctuation test framework:"

print "  =================================== "
print "	 First test: dataset with punctuation"
print "  =================================== "
print "  Data: practice, test, toy dataset"
print "  Features:  all features"
print "  Labels:    original 7-point scale"
print "  Model:     logistic regression (classification)"
print

corr, alpha, conf_matrix, details = experiment_frameworks.experiment_features_iterated(
    train_reader=icgauge.data_readers.punctuated_set, 
    assess_reader=None, 
    train_size=0.7,
    phi_list=[icgauge.feature_extractors.all_features], 
    class_func=icgauge.label_transformers.identity_class_func,
    train_func=icgauge.training_functions.fit_logistic_at,
    score_func=scipy.stats.stats.pearsonr,
    verbose=False,
    iterations=10)


print "\n-- AFTER COMPLETION --"
print "Averaged correlation (95% CI): " 
print np.round(np.mean(corr),2), "+/-", np.round(np.std(corr),2)
print "All correlations:"
print corr
print
print "Averaged Cronbach's alpha (95% CI): " 
print np.round(np.mean(alpha),2), "+/-", np.round(np.std(alpha),2)
print "All alphas:"
print alpha
print
print "Confusion matrix:"
print conf_matrix


"""
print "  =================================== "
print "	 Second test: dataset without punctuation"
print "  =================================== "
print "  Data: practice, test, toy dataset"
print "  Features:  all features"
print "  Labels:    original 7-point scale"
print "  Model:     logistic regression (classification)"
print

corr, alpha, conf_matrix, details = experiment_frameworks.experiment_features_iterated(
    train_reader=icgauge.data_readers.unpunctuated_set, 
    assess_reader=None, 
    train_size=0.7,
    phi_list=[icgauge.feature_extractors.all_features], 
    class_func=icgauge.label_transformers.identity_class_func,
    train_func=icgauge.training_functions.fit_logistic_at,
    score_func=scipy.stats.stats.pearsonr,
    verbose=False,
    iterations=10)


print "\n-- AFTER COMPLETION --"
print "Averaged correlation (95% CI): " 
print np.round(np.mean(corr),2), "+/-", np.round(np.std(corr),2)
print "All correlations:"
print corr
print
print "Averaged Cronbach's alpha (95% CI): " 
print np.round(np.mean(alpha),2), "+/-", np.round(np.std(alpha),2)
print "All alphas:"
print alpha
print
print "Confusion matrix:"
print conf_matrix
"""