# -*- coding: utf-8 -*-
#!/usr/bin/python

# Dev/Train/Test should be separated into different files
# with corresponding names

import json
import codecs
import numpy as np
import math

def read_format(src_filenames):
    """Iterator for cognitive complexity data.  The iterator 
    yields (paragraph, parse, label) pairs. 

    The labels are integers or numpy.nan. They are valid on an ordinal
    scale: 1 is less than 2, 2 is less than 3, but the distance between
    any two successive numbers is not necessarily valid.  Any transformation
    of these numbers occurs outside this function.
    
    Parameters
    ----------
    src_filenames : list of str
        List of full paths to the files to be read.

    Yields
    ------
    (paragraph, parse, label) as
        str, 
        list of strings parsable with Tree.fromstring(str), 
        value in {1,2,3,4,5,6,7,np.nan,None}, where:
            `np.nan` indicates "unscoreable" paragraphs, and
            `None` indicates paragraphs without a human assessment
    
    """
    for src_filename in src_filenames:
        print src_filename
        for item in json.load(codecs.open(src_filename, 'r', 'utf8')):
            paragraph = item["paragraph"]
            parse = None
            if "parse" in item.keys():
                parse = item["parse"]
            # parse = item["parse"] or None
            score = None
            if "score" in item:
                score = item["score"]
                if score == "NA":
                    score = np.nan
                elif isinstance(score, float):
                    score = int(math.floor(score + 0.5))
                assert score in [1,2,3,4,5,6,7,np.nan]
            yield (paragraph, parse, score)

def toy():
    """ Returns a reader for the toy data """
    return read_format(["sample_data/toy.json"])
    
def unscorable():
    """ Returns a reader for the unscoreable data """
    return read_format(["sample_data/unscorable.json"])

def train():
    """ Returns a reader for the combined train data """
    return read_format(["data/train.json"])

def dev():
    """ Returns a reader for the combined dev data """
    return read_format(["data/dev.json"])

def train_and_dev():
    """ Returns a reader for the train + dev data """
    return read_format(["data/train.json", "data/dev.json"])

def test():
    """ Returns a reader for the combined test data """
    return read_format(["data/test.json"])

def test_official():
    """ Returns a reader for the combined test data """
    return read_format(["data/test/test1.json_parsed.json"])

def test_obama():
    """ Returns a reader for Obama/McCain """
    return read_format(["data/test/politicalrhetoric_complexitycoding.json_parsed.json"])

def train_practice():
   """ Returns a reader for Practice Sets """
   return read_format(["data/train/practice_sets1-10_final_second_full_test.json_parsed.json"])

def train_oc():
   """ Returns a reader for Christian """
   return read_format(["data/train/OC_project_final.json_parsed.json"])

def train_heritability():
   """ Returns a reader for Reaction/Heritability """
   return read_format(["data/train/reaction_time_study_FINAL_second_full_test.json_parsed.json"])

def train_nixon():
   """ Returns a reader for Nixon """
   return read_format(["data/train/complexity_nixon_kennedy_FINAL.json_parsed.json"])


def train_applause():
   """ Returns a reader for Applause/Primaries """
   return read_format(["data/train/applause_second_full_test.json_parsed.json"])

def dev_bush():
    """ Returns a reader for Bush/Kerry """
    return read_format(["data/dev/bush_kerry_final_autoic.json_parsed.json"])

def punctuated_set():
    """ Returns a reader for the punctuated dataset: practice set, test set
    and toy dataset"""
    return read_format(["sample_data/practice_sets1-10_final_punctuated.json_parsed.json", 
        "sample_data/test1_punctuated.json_parsed.json", 
        "sample_data/toy.json"])
    
def unpunctuated_set():
    """ Returns a reader for the unpunctuated dataset: practice set, test set
    and toy dataset"""
    return read_format(["sample_data/practice_sets1-10_final_second_full_test.json_parsed.json",
        "sample_data/test1.json_parsed.json",
        "sample_data/toy_unpunctuated.json_parsed.json"])