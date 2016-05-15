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

import icgauge
from icgauge import experiment_frameworks


print "Toy framework:"
print "  Data:      toy.json"
print "  Features:  manual content flags"
print "  Labels:    original 7-point scale"
print "  Model:     logistic regression (classification)"
print

experiment_frameworks.experiment_features(
    train_reader=icgauge.data_readers.toy, 
    assess_reader=None, 
    train_size=0.7,
    phi_list=[icgauge.feature_extractors.manual_content_flags,
              icgauge.feature_extractors.pujun], 
    class_func=icgauge.label_transformers.identity_class_func,
    train_func=icgauge.training_functions.fit_maxent_with_crossvalidation,
    score_func=icgauge.utils.safe_weighted_f1,
    verbose=True)