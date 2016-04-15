# -*- coding: utf-8 -*-
#!/usr/bin/python

from collections import Counter
from nltk.tokenize import word_tokenize

import utils

# Note: Best to use independent namespaces for each key,
# since multiple feature functions can be grouped together.

def manual_content_flags(paragraph):
    """
    Baseline feature extractor, based on manual. Produces a feature function 
    that detects the presence of the example "content flag" phrases in the 
    coding manual.
    
    Parameters
    ----------
    paragraph : string
        Content string from which features should be extracted.
              
    Returns
    -------
    dict : string -> integer
        The number of times a flag at each level occurred in the text 
        (e.g., {"flag_1": 3, "flag_2": 1} )
      
    """
    flags = {"flag_1": utils.get_manual_flags(1),
             "flag_2": utils.get_manual_flags(2),
             "flag_3": utils.get_manual_flags(3),
             "flag_4": utils.get_manual_flags(4),
             "flag_5": utils.get_manual_flags(5),
            } 
    
    feature_presence = Counter()
    tokenized_and_lowercase = word_tokenize(paragraph.lower())
    for label, set_of_flags in flags.iteritems():
        for flag in set_of_flags:
            if flag in tokenized_and_lowercase:
                feature_presence[label] += 1

    return feature_presence
    
def unigrams(paragraph):
    """Produces a feature function on unigrams."""
    return Counter(word_tokenize(paragraph))
