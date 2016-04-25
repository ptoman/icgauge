# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
Usage:
From top level (/path/to/repo/icgauge), issue the command 
     `python -m experiments.simple`

These are very simple string-matching features,
on par with unigrams in terms of complexity
"""

import scipy
import numpy as np

import icgauge
from icgauge import experiment_frameworks


print "Simple framework:"
print "  Data:      toy.json"
print "  Features:  manual content flags, modals, length,"
print "             hedges, conjunctives, punctuation"
print "  Labels:    original 7-point scale"
print "  Model:     ordinal logistic model, all-threshold variant"
print "             (ordinal classification, recommended based on"
print "             experiments reported in Rennie and Srebro 2005"
print "             and replicated with toy.json)"
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
              icgauge.feature_extractors.punctuation_presence], 
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
             hedges, conjunctives, punctuation
  Labels:    original 7-point scale
  Model:     logistic regression (classification)

  ==================
  == Conclusions: ==
  ==================
  -- Big variance based on the split we happen to get
  -- A substantial gain from ordinal regression in this data-limited 
     situation: theory helps compensate for lack of data
  -- Minimal differences between the different ordinal methods, but
     LogisticAll-Threshold is recommended in literature and may have 
     slight advantage on this data

  == MaxEnt ==  
    0.287430554764
    [0.32951276778634803, 0.27360674099055743, 0.106834659486113, 0.27338029749597464, 0.45381830805918316]
    [[14  0  3  0  5  0  1]
     [ 7  3  0  0  0  0  0]
     [ 5  0  7  1  4  0  0]
     [ 4  0  1  0  5  0  0]
     [ 1  0  1  1  1  1  1]
     [ 3  0  2  0  0  0  5]
     [ 3  0  3  0  1  0  1]]

  == LogisticIT ==
    0.558938989521
    [0.56278859673371251, 0.44910377490990738, 0.41921490166815167, 0.78780180987912318, 0.73869207801138792, 0.51828012826474978, 0.51813215338239327, 0.58154956864450791, 0.62628825352759043, 0.3875386301893346]
    [[43  8 11  2  4  3  0]
     [12  0  8  0  4  0  0]
     [15  2 13  2  6  0  0]
     [ 0  5  4  1  5  0  1]
     [ 6  1 15  3  4  2  0]
     [ 0  1  4  0  3  2  4]
     [ 1  0  2  0  5  4  4]]
     
  == LogisticAT ==
    0.642424736644
    [0.7580429333814116, 0.60145708645926121, 0.70096154366588714, 0.7974983215002478, 0.43809737360231821, 0.61624663775277533, 0.60593703008382649, 0.78646410606992512, 0.43672983392124765, 0.68281250000000016]
    [[37 10  9  2  3  0  0]
     [11  4  7  1  0  0  0]
     [12  7  8 10  5  1  0]
     [ 2  0  5  3  3  1  0]
     [ 4  2  5  2  7  2  0]
     [ 0  0  1  3  1  1  3]
     [ 1  1  1  0  3  3  8]]
     
  == Ordinal Ridge Regression ==
    0.413543392401
    [0.15867235184970355, 0.77181119417950517, 0.42071738940549996, 0.46071473225427567, 0.28364627429757455, 0.3983883816732639, 0.64247677995116115, 0.59919085300459252, 0.087841996784177048, 0.31197397061228077]
    [[ 0  0  0  0  0  0  0  0]
     [10 12 21 10  8  4  3  2]
     [ 2  4  6  8  1  3  0  0]
     [ 7  8  9  8  6  4  2  3]
     [ 0  3  4  1  2  3  4  1]
     [ 1  1  7  4  2  4  0  1]
     [ 0  0  0  3  4  3  2  2]
     [ 1  1  0  2  2  4  2  5]]
  
"""

