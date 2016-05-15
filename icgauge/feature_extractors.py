# -*- coding: utf-8 -*-
#!/usr/bin/python

from collections import Counter
import re
import os

import numpy as np
from sklearn.decomposition import PCA

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import Tree

import utils
import utils_wordlists
import utils_parsing
import utils_similarity


GLOVE_SIZE = 50 # 50 is appropriate -- bigger starts to seriously 
                # outpace words per paragraph
path_to_glove = os.environ.get('GLV_HOME')
if not path_to_glove:
  raise ImportError("GLV_HOME not defined as environmental variable")
glove = None

# Note: Best to use independent namespaces for each key,
# since multiple feature functions can be grouped together.

# # for spell-checking
# import re, collections

# def words(text):
#     return re.findall('[a-z]+', text.lower())

# def train(features):
#     model = collections.defaultdict(lambda: 1)
#     for f in features:
#         model[f] += 1
#     return model

# NWORDS = train(words(file('big.txt').read()))
# alphabet = 'abcdefghijklmnopqrstuvwxyz'

# def edits1(word):
#     s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
#     deletes    = [a + b[1:] for a, b in s if b]
#     transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
#     replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
#     inserts    = [a + c + b     for a, b in s for c in alphabet]
#     return set(deletes + transposes + replaces + inserts)

# def known_edits2(word):
#     return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

# def known(words):
#     return set(w for w in words if w in NWORDS)

# def correct(word):
#     candidates = known([word]) or known(edits1(word)) or    known_edits2(word) or [word]
#     return max(candidates, key=NWORDS.get)

def word_length_features(paragraph):
    """
    Return a counter containing:
      avg_word_length
      median_frequency_word_length
      num_words_greater_than_6

    Parameters
    ----------
    paragraph : string
        Content string from which features should be extracted.

    Returns
    -------
    dict : string -> integer
    """
    features = Counter()

    features['avg_word_length'] = get_avg_word_length(paragraph)
    features['median_frequency'] = get_median_frequency(paragraph)
    features['num_words_greater_than_6'] = get_num_words_greater_than_x(paragraph, 6)

    return features

def get_avg_word_length(paragraph):
    """
    Return the average length of words
    """
    total_length = 0
    tokenized_and_lowercase = word_tokenize(paragraph.lower())
    for w in tokenized_and_lowercase:
      total_length += len(w)

    return total_length / len(tokenized_and_lowercase)

def get_median_frequency(paragraph):
    """
    Return median length after sorting the words based on their length.
    """
    tokenized_and_lowercase = word_tokenize(paragraph.lower())
    sorted_word_length_list = sorted([len(w) for w in tokenized_and_lowercase])
    if len(sorted_word_length_list) % 2 == 1:
      return sorted_word_length_list[len(sorted_word_length_list)/2]
    return (sorted_word_length_list[len(sorted_word_length_list)/2 - 1] + sorted_word_length_list[len(sorted_word_length_list)/2])/2

def get_num_words_greater_than_x(paragraph, x):
    """
    return the number of words that have more than x characters
    """
    tokenized_and_lowercase = word_tokenize(paragraph.lower())
    filtered_list = [w  for w in tokenized_and_lowercase if len(w) > x]
    return len(filtered_list)

def manual_content_flags(paragraph, unused_parse):
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
    flags = {"flag_1": utils_wordlists.get_manual_flags(1),
             "flag_2": utils_wordlists.get_manual_flags(2),
             "flag_3": utils_wordlists.get_manual_flags(3),
             "flag_4": utils_wordlists.get_manual_flags(4),
             "flag_5": utils_wordlists.get_manual_flags(5),
            } 
    
    feature_presence = Counter()
    tokenized_and_lowercase = word_tokenize(paragraph.lower())
    for label, set_of_flags in flags.iteritems():
        for flag in set_of_flags:
            if flag in tokenized_and_lowercase:
                feature_presence[label] += 1

    return feature_presence
    
def unigrams(paragraph, unused_parse):
    """Produces a feature function on unigrams."""
    return Counter(word_tokenize(paragraph))

def length(paragraph, unused_parse):
    """
    Produces length-related features. Rocket goes to the predictive
    performance of just this feature on the toy data (correlation between it
    and the human scores):
        - number of characters ==> 0.61 correlation (good feature)
        - number of white-space separated words (tokens) => 0.59
        - mean length of white-space separated tokens => 0.13
    """
    tokens = word_tokenize(paragraph)
    result = Counter()
    result["length_in_characters"] = len(paragraph)
    result["length_in_words"] = len(tokens)
    result["length_mean_word_len"] = np.mean([len(t) for t in tokens])
    return result
    
def wordlist_presence(wordlist_func, paragraph):
    """
    Produces feature list that is:
        - count of each word in the list generated by wordlist_func
        - count of tokens
        - count of types
        - proportion of tokens in paragraph that are in that list
    using a lower-case version of the original paragraph and a version of
    the paragraph in which all the tokens are separated by spaces (to 
    address cases where morphological forms are attached -- e.g.,
    "wouldn't" --> "would n't")
    """
    presence = Counter()
    paragraph = paragraph.lower()
    reconstituted_paragraph = " ".join(word_tokenize(paragraph))
    
    wordlist = wordlist_func()
    for phrase in wordlist:
        matcher = re.compile(r'\b({0})\b'.format(phrase), flags=re.IGNORECASE)
        matches = matcher.findall(paragraph)
        if len(matches) > 0:
            presence[phrase] += len(matches)
        else:
            matches = matcher.findall(reconstituted_paragraph)
            if len(matches) > 0:
                presence[phrase] += len(matches)
    
    return presence

def modal_presence(paragraph, unused_parse):
    modals = wordlist_presence(utils_wordlists.get_modals, paragraph)
    tokens = word_tokenize(paragraph)
    
    modals["modal_count_token"] = np.sum( \
        [value for key, value in modals.items()])    
    modals["modal_count_type"] = len(modals) - 1 # -1 bc *_count_token
    modals["modal_freq"] = 1.0*modals["modal_count_token"] / len(tokens)
    
    return modals
    
def hedge_presence(paragraph, unused_parse):
    hedges = wordlist_presence(utils_wordlists.get_hedges, paragraph)
    hedges["hedge_count_token"] = np.sum( \
        [value for key, value in hedges.items()])    
    hedges["hedge_count_type"] = len(hedges) - 1 # -1 bc *_count_token
    return hedges
    
def conjunctives_presence(paragraph, unused_parse):
    conjunctives = wordlist_presence(utils_wordlists.get_conjunctives, paragraph)
    conjunctives["conjunctive_count_token"] = np.sum( \
        [value for key, value in conjunctives.items()])    
    conjunctives["conjunctive_count_type"] = len(conjunctives) - 1 # -1 bc *_count_token
    return conjunctives

def punctuation_presence(paragraph, unused_parse):
    punctuation = utils_wordlists.get_punctuation()
    tokens = word_tokenize(paragraph.lower())
    
    result = Counter()
    for mark in punctuation:
        ct = tokens.count(mark)
        if ct > 0:
            result[mark] = ct
            
    result["punctuation_count_token"] = np.sum( \
        [value for key, value in result.items()])    
    result["punctuation_count_type"] = len(result) - 1 # -1 bc *_count_token
    result["punctuation_freq"] = 1.0*result["punctuation_count_token"] / len(tokens)
    
    return result

def determiner_usage(paragraph, parse, verbose=False):
  """
  Gets the count of times determiners are used per context.

  Central insight:
     Differences in determiner usage reflect whether the author assumes
     the noun phrase is a category that is coherent within common knowledge. 
     For instance:
        * We observe _the depravity_ of our age
        * Poverty, hunger, mental illness - they were _the inevitable result_ 
          of life in this world.
        * the Reagan administration is testing _the gullibility_ of world opinion
        * Abortion threatens _the moral and Christian character_ of this nation
     In all of these (pulled from low-complexity paragraphs), the author
     is assuming/presupposing a state of the world without having established its truth.
     Note that the presence of a detreminer in the subject doesn't have this
     meaning as often, because of the discourse rule that subjects
     are almost always *already* shared common knowledge -- we would expect subjects
     to often begin with "the" because they almost always have been introduced in 
     a prior sentence.

  Returns:
    dict, potentially with keys of:
       old info (for # times the determiner is in the subject,
             e.g., "The man wore a hat" -- the man was probably already 
             introduced earlier in the paragraph)
       knowledge assumed (for # times the determine assumes facts,
             e.g., "Some man wore the hat" -- err, what hat?)
       SBAR (for # times a determiner was accompanied by an SBAR, 
             e.g., "I saw the man wearing the hat" -- the man both assumed 
             and explained in terms of the knowledge assumed, "the hat")

  Notes:
    - Dependency parse may be better (cleaner and more accurate)
    - Might benefit from excluding proper noun phrases like country names
  """
  DETERMINER_LIST = ["the", "The"]
  features = Counter()
  for t_string in parse:
    t = Tree.fromstring(t_string)
    sent_not_shown = True
    for pos in t.treepositions('postorder'):
      if t[pos] in DETERMINER_LIST:
        phrase_of_interest = " ".join(t[pos[:-2]].leaves())
        while len(pos):
          match = utils_parsing.check_for_match(t, pos)
          if match:
            features["determiner_"+match] += 1
            if verbose:
              if sent_not_shown:
                print " ".join(t.leaves())
                sent_not_shown = False
              print "'%s' -- %s" % (phrase_of_interest, match)
            break
          pos = pos[:-1]
  return features

def dimensional_decomposition(paragraph, unused_parse, num_dimensions_to_accumulate=5):
  """ Gets the extent to which the word embeddings used in a paragraph
  can be reduced to to low-dimensional space.  Low-dimensional space is
  derived by PCA.

  Central insight:
     If the words all lay in a low-dimensional space, then likely
     they are expressing variations on the same theme rather 
     than nuanced argument.  Confounding variable of paragraph length
     actually seems good here.

  Empirical results:
     This works, and it captures something about content rather than
     just matching on highly functional words and phrases.
     On toy dataset, the first dimension has a correlation of -0.51
     with the score -- this is extremely encouraging!  The second-fifth
     cumulative dimensions are even stronger, all at about -0.56.

     Best to use a smaller number of dimensions (like 50), else we risk
     a highly overdetermined system of equations when heading into PCA.

  Returns:
     dictionary, with keys:
        cum_pca_var_expl_# (where # is in range 
           [0, num_dimensions_to_accumulate)); each value gives the amount
           of variance explained when that number of dimensions 
           is considered
  """
  global glove
  if glove == None:
    glove = utils.glove2dict(os.path.join(path_to_glove, 
            'glove.6B.%dd.txt' % GLOVE_SIZE))

  approx_num_words = int(1.5*len(paragraph.split()))
  word_vector = np.zeros((approx_num_words, GLOVE_SIZE))
  row = 0
  words = []
  for sent in sent_tokenize(paragraph):
    for word in word_tokenize(sent):
      word = word.lower()
      if word in glove and word not in words:
        word_vector[row] = glove[word]
        words.append(word)
        row += 1
  word_vector = word_vector[:row,:]
  #print words

  pca = PCA()
  pca.fit(word_vector)

  features = Counter()
  for i in range(num_dimensions_to_accumulate):
    features["cum_pca_var_expl_%d" % i] = np.sum(pca.explained_variance_ratio_[:i+1])

  return features

def syntactic_parse_features(paragraph, parse):
  """ Returns the count for the usage of S, SBAR units in the syntactic parse,
  plus statistics about the height of the trees  """
  KEPT_FEATURES = ['S', 'SBAR']

  # Increment the count for the part-of-speech of each head of phrase
  counts_of_heads = Counter()
  tree_heights = []
  for t_string in parse:  
    t = Tree.fromstring(t_string)
    for st in t.subtrees():
      counts_of_heads[st.label()] += 1
    tree_heights.append(t.height())

  # Keep only the head parts-of-speech that appear in KEPT_FEATURES
  features = dict(("syntactic_head_"+key, counts_of_heads[key]) for 
    key in counts_of_heads if key in KEPT_FEATURES)
  features = Counter(features)
  # Add in the features related to tree height
  features["tree_height_mean"] = np.mean(tree_heights)
  features["tree_height_median"] = np.median(tree_heights)
  features["tree_height_max"] = np.max(tree_heights)
  features["tree_height_min"] = np.min(tree_heights)
  features["tree_height_spread"] = np.max(tree_heights) - np.min(tree_heights)
  return features

def kannan_ambili(paragraph, parse):
  """ Semantic coherence as in Kannan-Ambili MS thesis at
  http://www.ai.uga.edu/sites/default/files/theses/KannanAmbili_aardra_2014Dec_MS.pdf

  This is the only feature in the existing literature designed specifically for 
  the integrative complexity task.

  Description:
    This metric uses a function of path length and depth of subsumer of
    word pairs (nouns and verbs only) in the WordNet hierarchy to generate 
    a self-similarity score for the paragraph.
    The score is based on the WordNet similarities between the words
    in the first sentence (assumed to be the topic sentence) and the
    other words in the paragraph.  See the thesis for more details.

  Empirical results:
    This feature correlates with toy data scores at 0.23.
  """
  trees = []
  for t_string in parse:
    trees.append(Tree.fromstring(t_string))
  if len(trees) < 2:
    return Counter()
  first_tree = trees[0]
  first_sentence_tokens = utils_parsing.get_nouns_verbs([first_tree])
  rest_trees = trees[1:]
  rest_sentences_tokens = utils_parsing.get_nouns_verbs(rest_trees) 

  similarities = np.zeros((len(first_sentence_tokens), len(rest_sentences_tokens)))
  for i, tuple1 in enumerate(first_sentence_tokens):
    for j, tuple2 in enumerate(rest_sentences_tokens):
      similarities[i,j] = utils_similarity.similarity_li(tuple1, tuple2)

  paragraph_semsim = np.exp(-np.sum(similarities.mean(axis=1)))

  return Counter({"kannan_ambili": paragraph_semsim})


# TODO:
# we need to get parses and keep them stored in parallel to the data --
# recreating the parses each time is killing the runtime

# Other potentially useful:
# - syntactic: passive voice
# - discourse: argument structure (statement-assessment as derived from but/and/because), 
#     sentiment both in terms of consistency and amount/strength, 
#     look up phrases in aphorism dictionary
# - lexical: most, more and other indicators of degree, misspellings
# - morphological: -ly on adjectives of degree, -est for superlatives
# - higher level: alignment of sentiment with determinism (draws on linguistic 
#     intergroup bias)
# - semantic: something compositional, something regarding argument structure
