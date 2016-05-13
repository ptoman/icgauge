# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet as wn

def get_synset(tuple_word_pos):
  """ Returns the most common synset for the word/pos pair `tuple_word_pos`, 
  or `None` if the word is unknown to WordNet """
  word, pos = tuple_word_pos
  synset = wn.synsets(word, pos)
  if len(synset) == 0:
    return None
  else:
    return synset[0]

def similarity_li(tuple1, tuple2, alpha=0.2, beta=0.6):
  """ Follows Li et al. 2003 as represented in Kannan Ambili 2014 thesis, page 41 """
  # Return 0 similarity if Noun+Verb pair (situation not specified in thesis)
  if tuple1[1] != tuple2[1]:
    return 0
  # Get synsets (approach not specified in thesis)
  synset1 = get_synset(tuple1)
  synset2 = get_synset(tuple2)
  # Return 0 similarity if word is not in wordnet (situation not specified in thesis)
  if synset1 is None or synset2 is None:
    return 0
  # Calculate path length, path depth
  path_length = 1 if synset1 == synset2 else synset1.shortest_path_distance(synset2)
  # If they aren't reachable from each other, no similarity btw words (not specified in thesis)
  if path_length is None:
    path_length = np.inf
  subsumers = synset1.lowest_common_hypernyms(synset2, simulate_root=True, 
    use_min_depth=True)
  # If nothing subsumes, there is no similarity between words (not specified)
  if len(subsumers) == 0:
    return 0
  subsumer = subsumers[0]
  path_depth = subsumer.max_depth() + 1
  # Calculate similarity 
  similarity = np.exp(-alpha * path_length) * \
                (np.exp(beta*path_depth) - np.exp(-beta*path_depth)) / \
                (np.exp(beta*path_depth) + np.exp(-beta*path_depth))
  return similarity

