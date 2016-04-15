# -*- coding: utf-8 -*-
#!/usr/bin/python


from sklearn.metrics import f1_score

def safe_weighted_f1(y, y_pred):
    """Weight-averaged F1, forcing `sklearn` to report as a multiclass
    problem even when there are just two classes. `y` is the list of 
    gold labels and `y_pred` is the list of predicted labels."""
    return f1_score(y, y_pred, average='weighted', pos_label=None)

def get_manual_flags(level="all"):
    """Returns a list of content flags from the qualitative coding manual
    available at http://www2.psych.ubc.ca/~psuedfeld/MANUAL.pdf.
    Valid `level` inputs are: ["all", 1, 2, 3, 4, 5].  Other inputs will
    return an empty list."""
    
    flags_for_1 = ["absolutely", "all", "always", "certainly", "constantly", 
                   "convinced", "definitely", "entirely", "forever", 
                   "impossible", "indisputable", "irrefutable", "irreversible", 
                   "never", "solely", "surely", "unconditionally", 
                   "undoubtedly", "unquestionably", "must", "clearly"]
    flags_for_2 = ["but", "nevertheless", "while", "however", "though", 
                   "probably", "almost", "usually", "may", "appears", "appear"]
    flags_for_3 = ["alternatively", "either", "on the other hand", "meanwhile"]
    flags_for_4 = ["it is likely that", "it seems possible", 
                   "they will probably", "likely", "possible", "possibly", 
                   "probably"]
    flags_for_5 = ["interplay", "interaction", "interdependency", "mutual", 
                   "mutuality", "mutually", "compromise", "equilibrium", 
                   "balancing", "trade-offs"]
    
    if level == "all":
        return flags_for_1 + flags_for_2 + flags_for_3 + flags_for_4 + flags_for_5
    elif level == 1:
        return flags_for_1
    elif level == 2:
        return flags_for_2
    elif level == 3:
        return flags_for_3
    elif level == 4:
        return flags_for_4
    elif level == 5:
        return flags_for_5
    else:
        return []