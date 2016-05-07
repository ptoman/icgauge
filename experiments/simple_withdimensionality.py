# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
Usage:
From top level (/path/to/repo/icgauge), issue the command 
     `python -m experiments.simple_withdimensionality`
"""

import scipy
import numpy as np

import icgauge
from icgauge import experiment_frameworks


print "Simple framework:"
print "  Data:      toy.json"
print "  Features:  manual content flags, modals, length,"
print "             hedges, conjunctives, punctuation, dimensionality reduction"
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
              icgauge.feature_extractors.dimensional_decomposition], 
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
  Data:      toy.json
  Features:  manual content flags, modals, length,
             hedges, conjunctives, punctuation, dimensionality reduction
  Labels:    original 7-point scale
  Model:     ordinal logistic model, all-threshold variant
             (ordinal classification, recommended based on
             experiments reported in Rennie and Srebro 2005)

  Iterated result: 
  0.629052902614
  [0.72231011192327133, 0.76246700400960943, 0.61149708229494548, 
   0.34068063698522627, 0.52518380387532992, 0.59206253525392449, 
   0.86733100375684946, 0.51148269367531829, 0.72958173743276455, 
   0.62793241693243074]
  [[30 11  7  1  2  1  2]
   [14  6  5  1  0  0  0]
   [ 9  3  5  2  4  1  0]
   [ 2  1  7  1  4  0  0]
   [ 1  3 14  4  5  0  0]
   [ 0  0  2  0  4  2  4]
   [ 1  0  1  0  2  0  6]]

  Second attempt:
  0.440007385885
  [0.64579744253822402, 0.34221919437031706, 0.0025711783922005163, 
   0.68460832060955412, 0.43730297426532866, 0.36632739877974291, 
   0.58292082373765675, 0.47942968277860643, 0.15331933882565046, 
   0.70557750455558621]
  [[28  6  4  3  7  2  0]
   [ 8  5  5  0  2  0  0]
   [10  4  6  2  4  0  0]
   [ 1  0  4  3  3  1  1]
   [ 1  0  8  3  1  2  0]
   [ 0  1  0  1  2  1  4]
   [ 3  0  1  0  2  4  4]]
 


  ==================
  == Conclusions: ==
  ==================
  -- May well help, especially given the nice correlation effect documented
     in the feature function (correlation of -0.56 with the eventual score).
     Hard to tell overall effect with the tiny little toy data, though, since
     splitting it up into train/test is challenging with this few examples.

"""

