# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
Usage:
From top level (/path/to/repo/icgauge), issue the command 
     `python -m experiments.simple`

These are very simple string-matching features,
on par with unigrams in terms of complexity
"""

import icgauge
from icgauge import experiment_frameworks


print "Simple framework:"
print "  Data:      toy.json"
print "  Features:  manual content flags, modals, length,"
print "             hedges, conjunctives, punctuation"
print "  Labels:    original 7-point scale"
print "  Model:     logistic regression (classification)"
print

experiment_frameworks.experiment_features(
    train_reader=icgauge.data_readers.toy, 
    assess_reader=None, 
    train_size=0.7,
    phi_list=[icgauge.feature_extractors.manual_content_flags,
              icgauge.feature_extractors.length,
              icgauge.feature_extractors.modal_presence,
              icgauge.feature_extractors.hedge_presence,
              icgauge.feature_extractors.conjunctives_presence,
              icgauge.feature_extractors.punctuation_presence], 
    class_func=icgauge.label_transformers.identity_class_func,
    train_func=icgauge.training_functions.fit_maxent_with_crossvalidation,
    score_func=icgauge.utils.safe_weighted_f1,
    verbose=True)
    
    
"""
Simple framework:
  Data:      toy.json
  Features:  manual content flags, modals, length,
             hedges, conjunctives, punctuation
  Labels:    original 7-point scale
  Model:     logistic regression (classification)

('Best params', {'penalty': 'l2', 'C': 2.0, 'fit_intercept': True})
Best score: 0.253
             precision    recall  f1-score   support

          1      0.455     0.833     0.588         6
          2      0.000     0.000     0.000         3
          3      0.500     0.167     0.250         6
          4      0.000     0.000     0.000         0
          5      0.000     0.000     0.000         4
          6      0.000     0.000     0.000         2

avg / total      0.273     0.286     0.239        21

[[5 0 0 1 0 0]
 [2 0 1 0 0 0]
 [1 2 1 1 1 0]
 [0 0 0 0 0 0]
 [3 0 0 1 0 0]
 [0 0 0 1 1 0]]
Correlation:  0.382997914546
(Rows are truth; columns are predictions)
"""
