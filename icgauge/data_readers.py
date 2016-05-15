# -*- coding: utf-8 -*-
#!/usr/bin/python

# Dev/Train/Test should be separated into different files
# with corresponding names

import json
import codecs
import numpy as np

def read_format(src_filename):
    """Iterator for cognitive complexity data.  The iterator 
    yields (paragraph, parse, label) pairs. 

    The labels are integers or numpy.nan. They are valid on an ordinal
    scale: 1 is less than 2, 2 is less than 3, but the distance between
    any two successive numbers is not necessarily valid.  Any transformation
    of these numbers occurs outside this function.
    
    Parameters
    ----------
    src_filename : str
        Full path to the file to be read.

    Yields
    ------
    (paragraph, parse, label) as
        str, 
        list of strings parsable with Tree.fromstring(str), 
        value in {1,2,3,4,5,6,7,np.nan,None}, where:
            `np.nan` indicates "unscoreable" paragraphs, and
            `None` indicates paragraphs without a human assessment
    
    """
    for item in json.load(codecs.open(src_filename, 'r', 'utf8')):
        paragraph = item["paragraph"]
        parse = item["parse"] or None
        score = None
        if "score" in item:
            score = item["score"]
            if score == "NA":
                score = np.nan
            assert score in [1,2,3,4,5,6,7,np.nan]
        yield (paragraph, parse, score)

def toy():
    """ Returns a reader for the toy data """
    return read_format("sample_data/toy.json")
    
def unscorable():
    """ Returns a reader for the unscoreable data """
    return read_format("sample_data/unscorable.json")
