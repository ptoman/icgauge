# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
This file contains experimental results related to the 
simple features model.

Usage:
From top level (/path/to/repo/icgauge), issue the command 
     `python -m experiments.simple_features`
"""


import json
from collections import defaultdict
from matplotlib import pyplot as plt

import scipy
import numpy as np

import icgauge
from icgauge import experiment_frameworks
from icgauge import feature_extractors

run_experiment = True

if run_experiment:
  corr, alpha, conf_matrix, details = experiment_frameworks.experiment_features_iterated(
      train_reader=icgauge.data_readers.train_and_dev, 
      assess_reader=icgauge.data_readers.test,#_official, 
      train_size=0.7,
      phi_list=[
                 icgauge.feature_extractors.simple_features
               ], 
      class_func=icgauge.label_transformers.identity_class_func, #ternary_class_func
      train_func=icgauge.training_functions.fit_logistic_at,#_with_crossvalidation,
      score_func=scipy.stats.stats.pearsonr,
      verbose=False,
      iterations=1)

  # Print out the results
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

  # Store the results to disk -- "truth"/"prediction"/"example"
  with open("results.json", "w") as fp:
      json.dump(details, fp, indent=4)


"""""""""""""""""""""
Experimental results.

Cross-validation on train:
 alpha = 2.0, 2.0, 3.0, 3.0, 2.0, 0.8, 0.6, 0.6, 0.2, 3.0
avg is 2.0 -- so we use 2.0

Test (full set -- not the official test):


-- AFTER COMPLETION --
Averaged correlation (95% CI): 
0.52 +/- 0.0
All correlations:
[0.51659610020010571]

Averaged Cronbach's alpha (95% CI): 
0.67 +/- 0.0
All alphas:
[0.66919834579290627]

Confusion matrix:
[[10 46  3  0  0  0  0]
 [ 5 57 12  0  1  0  0]
 [ 0 22 19  0  1  2  0]
 [ 0  0  4  2  0  1  0]
 [ 0  3  1  1  0  0  0]
 [ 0  0  1  0  0  0  0]
 [ 0  0  0  0  0  1  0]]


"""""""""""""""""""""

