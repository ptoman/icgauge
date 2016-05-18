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


corr, conf_matrix, details = experiment_frameworks.experiment_features_iterated(
    train_reader=icgauge.data_readers.train, 
    assess_reader=None, 
    train_size=0.7,
    phi_list=[icgauge.feature_extractors.manual_content_flags,
              icgauge.feature_extractors.length,
              icgauge.feature_extractors.word_length_features,
              icgauge.feature_extractors.modal_presence,
              icgauge.feature_extractors.get_more_most_counts,
              icgauge.feature_extractors.hedge_presence,
              icgauge.feature_extractors.get_morphological_counts,
              icgauge.feature_extractors.transitional_presence,
              icgauge.feature_extractors.conjunctives_presence,
              icgauge.feature_extractors.punctuation_presence,
              icgauge.feature_extractors.determiner_usage,
              icgauge.feature_extractors.dimensional_decomposition,
              icgauge.feature_extractors.syntactic_parse_features,
              icgauge.feature_extractors.kannan_ambili
             ], 
    class_func=icgauge.label_transformers.identity_class_func, #vs. ternary_class_func
    train_func=icgauge.training_functions.fit_logistic_at_with_crossvalidation,  # does not have crossvalidation
    score_func=scipy.stats.stats.pearsonr,
    verbose=False,
    iterations=1)

# Print out the results
print "\n-- AFTER COMPLETION --"
print "Averaged correlation: " 
print np.mean(corr)
print "All correlations:"
print corr
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

  
"""

