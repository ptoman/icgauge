# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
Usage:
From top level (/path/to/repo/icgauge), issue the command 
     `python -m experiments.all_features`
"""

import cProfile
import json

import scipy
import numpy as np

import icgauge
from icgauge import experiment_frameworks


corr, alpha, conf_matrix, details = experiment_frameworks.experiment_features_iterated(
    train_reader=icgauge.data_readers.train_and_dev, 
    assess_reader=icgauge.data_readers.test, 
    train_size=0.7,
    phi_list=[icgauge.feature_extractors.all_features
             ], 
    class_func=icgauge.label_transformers.identity_class_func, #vs. ternary_class_func
    train_func=icgauge.training_functions.fit_logistic_at_6,#with_crossvalidation,
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


"""
Using all feature functions.

  ==================
  == Conclusions: ==
  ==================

- Trying all alphas:
   6.0 or 8.0, 8.0, 8.0, 6.0, 4.0, 3.0, 6.0, 4.0, 4.0
    

---------------------------
Test results
















------------------------------
Test official






















----------------------------
Obama/McCain
  
"""

