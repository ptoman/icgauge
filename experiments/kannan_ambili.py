# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
This file contains experimental results related to the 
Kannan-Ambili semantic coherence metric.

Usage:
From top level (/path/to/repo/icgauge), issue the command 
     `python -m experiments.kannan_ambili`
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
                 icgauge.feature_extractors.semcom_ka_features
               ], 
      class_func=icgauge.label_transformers.identity_class_func, #ternary_class_func
      train_func=icgauge.training_functions.fit_logistic_at,#_with_crossvalidation,
      score_func=scipy.stats.stats.pearsonr,
      verbose=False,
      iterations=5)

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
 alpha = 1.0, 0.2, 0.2, 0.2, 0.4, 0.8, 0.2, 2.0, 0.2, 0.4, 0.8
If only using best: 0.28 and 0.17
Try 0.6 -- it's closest to average -- and get 0.22 and 0.15
Try 0.2 -- and get 0.25 and 0.15
... these are basically the same. so let's stick with 
the average ==> 0.6.


Test:
0 correlation, 0 alpha

Test official:
0 correlation, 0 alpha

"""""""""""""""""""""

