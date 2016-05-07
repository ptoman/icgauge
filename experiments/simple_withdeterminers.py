# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
Usage:
From top level (/path/to/repo/icgauge), issue the command 
     `python -m experiments.simple_withdeterminers`
"""

import scipy
import numpy as np

import icgauge
from icgauge import experiment_frameworks


print "Simple framework:"
print "  Data:      toy.json"
print "  Features:  manual content flags, modals, length,"
print "             hedges, conjunctives, punctuation, determiner_usage"
print "  Labels:    original 7-point scale"
print "  Model:     ordinal logistic model, all-threshold variant"
print "             (ordinal classification, recommended based on"
print "             experiments reported in Rennie and Srebro 2005)"
print


corr, conf_matrix = experiment_frameworks.experiment_features_iterated(
    train_reader=icgauge.data_readers.toy, 
    assess_reader=None, 
    train_size=0.7,
    phi_list=[icgauge.feature_extractors.manual_content_flags,
              icgauge.feature_extractors.length,
              icgauge.feature_extractors.modal_presence,
              icgauge.feature_extractors.hedge_presence,
              icgauge.feature_extractors.conjunctives_presence,
              icgauge.feature_extractors.punctuation_presence,
              icgauge.feature_extractors.determiner_usage], 
    class_func=icgauge.label_transformers.identity_class_func,
    train_func=icgauge.training_functions.fit_logistic_at_with_crossvalidation,
    score_func=scipy.stats.stats.pearsonr,
    verbose=False,
    iterations=10)

print "Iterated result: " 
print np.mean(corr)
print corr
print conf_matrix
print np.sum(conf_matrix)

"""
Simple framework:
  0.492744888593
  [0.44064985793404876, 0.41152744587655088, 0.36572957483083135, 
   0.52459551865236986, 0.67763117511199777, 0.68270993548331238, 
   0.54365402738555813, 0.7211953848183027, 0.42287058530853344, 
   0.13688538052878968]
  [[35 14 16  2  4  2  0]
   [ 8  5  3  2  1  0  0]
   [ 6  5  7  3  6  2  0]
   [ 3  4  4  2  4  2  0]
   [ 4  5  6  3  3  1  0]
   [ 0  0  0  0  4  2  3]
   [ 4  0  1  1  1  9  2]]


  ==================
  == Conclusions: ==
  ==================

  -- *So* *freaking* *slow* to use the determiners feature -- it's all 
     the parsing plus inefficient looping code on top. If this feature proves
     useful, this is an area for algorithm performance improvement.
  -- Hard to tell if this is actually useful. It falls right in between
     where our previous values fall.


"""

