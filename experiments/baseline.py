# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
This file contains experimental results related to the baseline.

Usage:
From top level (/path/to/repo/icgauge), issue the command 
     `python -m experiments.baseline`
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
      assess_reader=icgauge.data_readers.test_official, 
      train_size=0.7,
      phi_list=[
                 icgauge.feature_extractors.baseline_features
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

* train + dev
* 7-point scale

Investigate the right parameters using crossvalidation first:
  alpha = 0.2, 0.2, 0.8, 0.4, 1.0, 0.4, 0.2, 0.2, 0.2, 0.2
On average: 0.4 is closest, but there isn't a ton of consistency here...


#######################
Test

Correlation
0.34

Cronbach's alpha
0.36

Confusion
[[ 1 55  3  0  0  0  0]
 [ 0 66  9  0  0  0  0]
 [ 0 32 12  0  0  0  0]
 [ 0  1  6  0  0  0  0]
 [ 0  4  1  0  0  0  0]
 [ 0  1  0  0  0  0  0]
 [ 0  0  1  0  0  0  0]]


#######################
Test - Official

Correlation
0.29

Cronbach's alpha
0.27

Confusion
[[0 5 0 0 0 0 0]
 [0 6 2 0 0 0 0]
 [0 5 2 0 0 0 0]
 [0 1 4 0 0 0 0]
 [0 3 0 0 0 0 0]
 [0 1 0 0 0 0 0]
 [0 0 1 0 0 0 0]]


"""""""""""""""""""""

