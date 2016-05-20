# -*- coding: utf-8 -*-
#!/usr/bin/python

"""
This file contains experimental results related to the PCA / 
lower-dimensional subspace conception of semantic complexity.

52% of the training data is less than 50 words, so this would
  be a very unrepresentative sample
19% of the training data is less than 30 words
8% of the training data is less than 20 words
0.6% of the training data is less than 5 words

Usage:
From top level (/path/to/repo/icgauge), issue the command 
     `python -m experiments.pca`
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
      assess_reader=None, #icgauge.data_readers.test_official, 
      train_size=0.7,
      phi_list=[
        # These two features were used to control for the effect of number
        # of words and to get at the effect of the decomposition itself
        #        icgauge.feature_extractors.number_words_only,
        #        icgauge.feature_extractors.dimensional_decomposition,
        # The real feature function for PCA, though, is:
                 icgauge.feature_extractors.semcom_pca_features
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
* non-cumulative values
* 50 eigenvalues

First qualitative standout -- we never predict above a 4, and 4s are very rare.
This might be lack of data to train the high end.

Cross-validation -- Do we have good parameters for Logistic AT?
  Tries alphas: [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0
  Best param = 0.2 always.
All further PCA experiments in this section use LogisticAt(alpha=0.2).

Model: LogisticAT is good! It's more useful even than weighting classes.
It might help the most to reweight and us LogisticAT -- but that 
requires reworking the mord ordinal regression library.
  LogisticAT
    0.30 +/ 0.02
  MaxEnt with weighted classes:
    0.24 +/- 0.02
  MaxEnt
    0.11 +/- 0.03


Features: When controlling for the number of words, we the gain from
adding dimensionality exists but borders on statisticaly insignificance...
So, this feature is likely mostly just capturing length.
  num words alone 
    0.30 +/- 0.05
  num words + dimensional
    0.36 +/- 0.04
  (dimensional alone)
    0.30 +/ 0.02

------------------------------------------


* train + dev
* 7-point scale
* cumulative values
* 20 eigenvalues

Cross-validation -- Do we have good parameters for Logistic AT?
  Tries alphas: [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0
  Best: 0.8, 0.6, 0.6, 0.2, 0.2, 0.2, 0.4, 0.2, 0.2
So, we use 0.4 (this is the value closest to the average)

Now we look at the performance using the cumulative setup with only 20 eigenvalues:
  num words + dimensional
    0.38 +/- 0.04
  dimensional alone
    0.39 +/ 0.03


------------------------------------------


* train + dev
* 7-point scale
* non-cumulative values
* 20 eigenvalues

Cross-validation -- Do we have good parameters for Logistic AT?
  Tries alphas: [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0
  Winner is consistently 0.2  This suggests that non-cumulative values really want alpha of 0.2.

Now we look at the performance using the non-cumulative setup with only 20 eigenvalues:
  num words + dimensional
    0.36 +/- 0.06
  dimensional alone
    0.29 +/- 0.04
This is a decrease compared to the top 50 non-cumulative eigenvalues, and both of those
are lower than the cumulative eigenvalues. This suggests that the models aren't able
to derive cumulativeness (of course), and that we're better off using a cumulative approach.

So, we turn to a cumulative approach and focus in on the number of eigenvalues.


------------------------------------------


* train + dev
* 7-point scale
* cumulative values
* 50 eigenvalues

Cross-validation -- Do we have good parameters for Logistic AT?
  Tries alphas: [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0
  Always prefers 0.2.

Now let's look at performance:
  dimensional alone
    0.39 +/- 0.04

Note that this is exactly the same as with 20 dimensions. So, let's 
check what happens using fewfer than 20 dimensions.

First, though, let's check extensibility.
The fact that the dimensional value goes up regularly is surprisingly....
I wonder if we're overfitting. Let's train on train and test on dev to see
how performance is affected.
  dimensional alone
    0.34
That's still reasonable.

------------------------------------------


* train + dev
* 7-point scale
* cumulative values
* 10 eigenvalues

Now let's look at performance:
  word alone
    0.33 +/- 0.06
  dimensional + num words
    0.39 +/- 0.03
  dimensional alone
    0.39 +/- 0.03
These are equivalent :( So can't say much about whether PCA is getting
more than just number of words.

Let's decrease num eigenvalues again.


------------------------------------------


* train + dev
* 7-point scale
* cumulative values
* 5 eigenvalues

Now let's look at performance:
  dimensional alone
    0.35 +/- 0.3
This is our first drop. This suggests that we only need somewhere between 
5 and 10 dimensions to predict scores.


------------------------------------------


* train + dev
* 7-point scale
* cumulative values
* 8 eigenvalues

Now let's look at performance:
  dimensional alone
    0.36 +/- 0.4
Yep, I'm thinking 10 dimensions is the one to embrace -- it's the lowest
that gets us the highest extrapolation score.


#################################################
# TEST PERFORMANCE
* train + dev --> test
* 7-point scale
* cumulative values
* 10 eigenvalues
* alpha = 0.2

Correlation:
0.38

Cronbach's alpha:
0.44

Confusion matrix:
[[ 5 47  7  0  0  0  0]
 [ 0 63 12  0  0  0  0]
 [ 0 26 18  0  0  0  0]
 [ 0  2  5  0  0  0  0]
 [ 0  3  2  0  0  0  0]
 [ 0  0  1  0  0  0  0]
 [ 0  0  1  0  0  0  0]]




Let's break this down onto TEST_OFFICIAL

Correlation:
0.47

Cronbach's alpha:
0.46

Confusion matrix:
[[2 3 0 0 0 0 0]
 [0 6 2 0 0 0 0]
 [0 5 2 0 0 0 0]
 [0 2 3 0 0 0 0]
 [0 3 0 0 0 0 0]
 [0 0 1 0 0 0 0]
 [0 0 1 0 0 0 0]]


#################################################

Now let's look at 3-class case.


Cross-validation -- Do we have good parameters for Logistic AT?
  Tries alphas: [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0
  0.4, 0.2, 0.2, 0.2, 0.2, 0.2, 0.6, 0.2, 0.2, 0.2
This is not quite as strongly in favor of 0.2 but still seems
happiest there.


Now let's look at performance:

* train + dev --> crossvalidation
* 3-point scale
* cumulative values
* 10 eigenvalues
* alpha = 0.2
Now we look at the performance using the non-cumulative setup with only 20 eigenvalues:
  num words + dimensional
    0.28 +/- 0.05
  dimensional alone
    0.27 +/- 0.03
This is much poorer performance than what we were seeing in the 7-class approach. 
This suggests that the 7-class task might be an easier one -- suggesting that the
data might well be 7-dimensional (or at least higher than 3-dimensional).
But we might as well report these results for consistency with the 7-class model, 
since there is zero reason to suggest that a different number of eigenvalues will
perform better for 3-class than for 7-class.

Let's try 20 eigenvalues instead of 10:
  dimensional alone
    0.33 +/- 0.04
It's basically the same.

Let's try 50 eigenvalues instead of 20:
  dimensional alone
    0.29 +/- 0.05
It's basically the same. 

Let's try for 30..
  dimensional alone
    0.29 +/- 0.02
It's basically the same.


Let's try for 5..
  dimensional alone
    0.20 +/- 0.04
It's definitely worse to only use 5.



#################################################
# TEST PERFORMANCE
* train + dev --> test
* 3-point scale
* cumulative values
* 10 eigenvalues
* alpha = 0.2

Correlation:
0.31

Cronbach's alpha:
0.47

Confusion matrix:
[[115  19   0]
 [ 32  24   0]
 [  1   1   0]]



Let's break this down onto TEST_OFFICIAL

Correlation:
0.12

Cronbach's alpha:
0.20

Confusion matrix:
[[10  3  0]
 [11  4  0]
 [ 1  1  0]]


#################################################

"""""""""""""""""""""




# Get the distribution of paragraph lengths
word_count = []
for paragraph, parse, label in icgauge.data_readers.train():
  num_words = len(paragraph.split())
  word_count.append(num_words)


# Collect PCA results for paragraphs of each size range
DIM_TO_ACCUM = 50
by_range = defaultdict(list)
by_score = defaultdict(list)
for paragraph, parse, label in icgauge.data_readers.train():
  # Figure out where this paragraph falls in terms of lengthiness
  num_words = len(paragraph.split())
  perc = scipy.stats.percentileofscore(word_count, num_words)
  centered_bin = np.ceil(perc/20)*20 - 10 # e.g, 5 means the percentile is in (0,10]

  # Get the PCA eigenvalues and padd to DIM_TO_ACCUM
  pca = feature_extractors.derive_pca_on_glove(paragraph, DIM_TO_ACCUM)
  #print np.round(pca.explained_variance_ratio_[0],2), label
  cols = pca.explained_variance_ratio_.shape[0]
  padded_eigenvalues = np.lib.pad(pca.explained_variance_ratio_, 
    (0, DIM_TO_ACCUM - cols), 
    'constant', 
    constant_values=0.)

  by_range[centered_bin].append(padded_eigenvalues)
  if centered_bin > 10 and centered_bin < 90:
    by_score[label].append(padded_eigenvalues)




plot_effect_of_length = False
plot_distro_by_score = False

if plot_effect_of_length:
  # Plot to check the effect of paragraph length on the distribution of PCA eigenvalues.
  # The result is that the top 20% and the bottom 20% follow a different distribution
  # and should be trimmed before plotting the ability of PCA eigenvalues to predict 
  # scores
  for key in range(10, 100, 20):
    distribution = by_range[key]
    as_matrix = np.vstack(distribution)

    mean = np.mean(as_matrix, axis = 0)
    std2 = 2*np.std(as_matrix, axis = 0)

    plt.plot(range(0,DIM_TO_ACCUM), mean, "-D", label=str(key))
    #plt.errorbar(range(0, DIM_TO_ACCUM), mean, yerr=std2, linestyle="None")
  plt.legend()
  plt.show()



if plot_distro_by_score:
  for key in range(1,8):
    distribution = by_score[key]
    if len(distribution) > 0:
      as_matrix = np.vstack(distribution)

      mean = np.mean(as_matrix, axis = 0)
      std2 = 2*np.std(as_matrix, axis = 0)

      plt.plot(range(0,DIM_TO_ACCUM), mean, "-D", label=str(key))  
  plt.legend()
  plt.show()
