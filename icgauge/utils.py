# -*- coding: utf-8 -*-
#!/usr/bin/python

from sklearn.metrics import f1_score

def safe_weighted_f1(y, y_pred):
    """Weight-averaged F1, forcing `sklearn` to report as a multiclass
    problem even when there are just two classes. `y` is the list of 
    gold labels and `y_pred` is the list of predicted labels."""
    return f1_score(y, y_pred, average='weighted', pos_label=None)
