# -*- coding: utf-8 -*-
#!/usr/bin/python

from nltk.tree import Tree
from nltk.parse.stanford import StanfordParser
from nltk.tokenize import sent_tokenize
import os

path_to_stanford_nlp = os.environ.get('STANFORD_NLP_HOME')
if not path_to_stanford_nlp:
  raise ImportError("STANFORD_NLP_HOME not defined as environmental variable")


english_parser = StanfordParser(os.path.join(path_to_stanford_nlp,'stanford-parser.jar'),
                   os.path.join(path_to_stanford_nlp, 'stanford-parser-3.4.1-models.jar'))

def get_trees_given_paragraph(paragraph):
  """ Yields the tree for each sentence """
  sentences = sent_tokenize(paragraph)
  return [tree for tree in get_trees(sentences)]

def get_trees(sentences):
  """ Yields the tree for each sentence """
  parsed_sentences = english_parser.raw_parse_sents(sentences)
  for i in parsed_sentences:
    yield list(i)[0]


def syntax_of_determiner_usage(paragraph, verbose=False):
  """ Produces the syntactic context of the use of each determiner
  in a paragraph.  Currently unused, but this could be helpful
  given enough data to populate all the possible category variants.

  Inputs:
    `paragraph`: string

  Returns:
    list of lists of tuples,
      For each determiner in the paragraph, returns a description of
      the parsing path from sentence (S) to determiner. Each left 
      tuple is a string representing the part of speech at that level, 
      and each right tuple is an integer representing whether that unit 
      is a left, middle, or right branch from its parent

  Example:
    "The Saudi oil policy may look inconsistent to outsiders, but 
     the appearance is misleading." -->
    [[(u'S', 0), (u'S', 0), (u'NP', 0), (u'DT', 0)],
     [(u'S', 0), (u'S', 3), (u'NP', 0), (u'DT', 0)]]

    Which arises from the following parse tree (not returned):
    (ROOT
      (S
        (S
          (NP (DT The) (JJ Saudi) (NN oil) (NN policy))
          (VP
             (MD may)
            (VP
              (VB look)
              (ADJP (JJ inconsistent) (PP (TO to) (NP (NNS outsiders)))))))
        (, ,)
        (CC but)
        (S
          (NP (DT the) (NN appearance))
          (VP (VBZ is) (ADJP (JJ misleading))))
        (. .)))
  """
  sentences = sent_tokenize(paragraph)
  paragraph_structure = []
  for t in get_trees(sentences):
    if verbose:
      print t
    for pos in t.treepositions('postorder'):
      if t[pos] == "the" or t[pos] == "The":
        parse_path = []
        while len(pos):
          try:
            current_unit = (t[pos].label(), pos[-1])
            parse_path.append(current_unit)
          except AttributeError:
            pass
          pos = pos[:-1]
        if verbose:
          print parse_path[::-1]
        paragraph_structure.append(parse_path[::-1])
  return paragraph_structure

def get_neighbor_pos(pos):
  """ Helper function to get right sibling -- no guarantee it exists in tree """
  neighbor = list(pos)
  neighbor[-1] += 1
  return neighbor

def check_for_match(t, pos):
  """ Helper function that returns if location `pos` in tree `t` is of type SBAR, 
  knowledge assumed, or old info.  If none, returns `None`."""
  try:
    label = t[pos].label()
    if label == "NP":
      neighbor = get_neighbor_pos(pos)
      try:
        if t[neighbor].label() == "SBAR":
          return "SBAR"
      except IndexError:
        pass
    elif label == "VP":
      return "knowledge assumed"
    elif label == "S":
      return "old info"
  except AttributeError:
    pass
  return None

def get_nouns_verbs(list_of_trees):
  """ Returns the noun and verb tokens in a sentence as tuples: (word, ['v'|'n']) """
  ALLOWABLE = ['NN', 'NNS', 'NNP', 'NNPS', 
               'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
  sentence_tokens = []
  for source_tree in list_of_trees:
    source_poses = source_tree.pos()
    for source_pos in source_poses:
      if source_pos[1] in ALLOWABLE:
        sentence_tokens.append((source_pos[0], source_pos[1][0].lower()))
  return sentence_tokens



