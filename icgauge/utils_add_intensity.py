#!/usr/bin/python

# This file appends parse trees to all the datasets if the data 
# do not currently have trees 
#
# Usage: python -m icgauge.utils_add_parse_trees

import os
import json
import numpy as np

from utils_vsm import *

so = semantic_orientation_score_lexicon(mat=glv[0], rownames=glv[1],
        negset=('little', 'few', 'never', 'low', 'negative', 'wrong', 'small', 'inferior', 'seldom'),
        posset=('extremely', 'many', 'always', 'high', 'positive', 'correct', 'big', 'superior', 'often'))

def get_intensity(paragraph):
    intensity = []
    sentence = []
    sentence_means = []
    for word in paragraph.split():
        if word in so:
            intensity.append(so[word])
            sentence.append(so[word])
        if word == ".":
            m = np.mean(sentence)
            sentence_means.append(m)
            sentence = []
    sentence_var = 0
    print "sentence means"
    print sentence_means
    if len(sentence_means) > 1:
        sentence_var = np.var(sentence_means)
    return {'all_var': np.var(intensity), 'sentence_var': sentence_var}

def add_parse_trees():
    for dirname in ["sample_data", "data"]:
        for fn in os.listdir(dirname):
            if fn.endswith(".json"):
                print "  Checking", fn
                with open(os.path.join(dirname,fn)) as json_file:
                    needs_new_file = False
                    dataset = json.load(json_file)
                    revised_items = []
                    for item in dataset:
                        if "parse" not in item:
                            needs_new_file = True
                            get_intensity(item['paragraph'])
                            print get_intensity(item['paragraph'])
                            item["intensity"] = 1
                        # revised_items.append(item)
                    # if needs_new_file:
                        # with open(os.path.join(dirname, fn+"_scored.json"), 'w') as to_file:
                            # json.dump(revised_items, fp=to_file, indent=4)

if __name__ == '__main__':
    add_parse_trees()
