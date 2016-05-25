# -*- coding: utf-8 -*-
#!/usr/bin/python

from collections import Counter
import re
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import Tree

import utils
import utils_wordlists
import utils_parsing
import utils_similarity
from utils_vsm import semantic_orientation_score_lexicon


GLOVE_SIZE = 50 # 50 is appropriate -- bigger starts to seriously 
                # outpace words per paragraph
path_to_glove = os.environ.get('GLV_HOME')
if not path_to_glove:
  raise ImportError("GLV_HOME not defined as environmental variable")
glove = None

semantic_lexicon = None
if os.path.exists(os.path.join('data/', 'semantic_lexicon.pickle')):
  semantic_lexicon = pickle.load(file(os.path.join('data/', 'semantic_lexicon.pickle')))

# Note: Best to use independent namespaces for each key,
# since multiple feature functions can be grouped together.

lstm_vocab = None
if os.path.exists(os.path.join('data/', 'lstm_vocab')):
  lstm_vocab = pickle.load(file(os.path.join('data/', 'lstm_vocab')))


lstm_representation_dict = None
if os.path.exists(os.path.join('data/', 'lstm_representation_dict')):
  lstm_representation_dict = pickle.load(file(os.path.join('data/', 'lstm_representation_dict')))

###########################################
# Master feature functions:
###########################################

def baseline_features(paragraph, parse):
  return manual_content_flags(paragraph, parse) | \
    baseline_length(paragraph, parse)

def simple_features(paragraph, parse):
  return lexical_features(paragraph, parse) | \
    length(paragraph, parse) | \
    syntactic_features(paragraph, parse)

def semcom_sentiment_features(paragraph, parse):
  return word_intensity(paragraph, parse)

def semcom_pca_features(paragraph, unused_parse):
  # As per experimental results, cumulative + 10 eigenvalues is best
  return dimensional_decomposition(paragraph, unused_parse, 10)

def semcom_lstm_features(paragraph, unused_parse):
  global lstm_representation_dict
  result = lstm_representation_dict[paragraph][0]
  return Counter({'lhs_prob': result[0], 'rhs_prob': result[1]})



def semcom_ka_features(paragraph, parse):
  return kannan_ambili(paragraph, parse)

def all_features(paragraph, parse):
  return baseline_features(paragraph, parse) | \
    lexical_features(paragraph, parse) | \
    length(paragraph, parse) | \
    syntactic_features(paragraph, parse) | \
    semcom_sentiment_features(paragraph, parse) | \
    semcom_pca_features(paragraph, parse) | \
    semcom_lstm_features(paragraph, parse) | \
    semcom_ka_features(paragraph, parse)

##########################################
# Sub-units (for ablation)
# - lexicographic features (= length)
# - lexical features (particular word categories)
# - syntactic features (determiners & parse complexity)
##########################################
def lexical_features(paragraph, parse):
  return get_morphological_counts(paragraph, parse) | \
    modal_presence(paragraph, parse) | \
    relative_amount_presence(paragraph, parse) | \
    transitional_presence(paragraph, parse) | \
    hedge_presence(paragraph, parse) | \
    conjunctives_presence(paragraph, parse) | \
    punctuation_presence(paragraph, parse)

def syntactic_features(paragraph, parse):
  return determiner_usage(paragraph, parse) | \
    syntactic_parse_features(paragraph, parse)


###########################################
# Components:
###########################################

def get_morphological_counts(paragraph, unused_parse):
  """
  Return a counter containing:
   number of times '-----er' occurs
   number of times '-----ly' occurs
   number of times '-----est' occurs

  Parameters
  ----------
  paragraph : string
      Content string from which features should be extracted.

  Returns
  -------
  dict : string -> integer
  """
  features = Counter()
  tokenized_and_lowercase = word_tokenize(paragraph.lower())
  er_count = 0
  ly_count = 0
  est_count = 0
  for w in tokenized_and_lowercase:  
    if w.endswith('er'):
      er_count += 1
    elif w.endswith("est"):
      est_count += 1
    elif w.endswith("ly"):
      ly_count += 1

  features['er_count'] = er_count
  features['est_count'] = est_count
  features['ly_count'] = ly_count

  return features


def word_intensity(paragraph, unused_parse):
    """
    Return level of word intensity in the paragraph based on semantic orientation.
    Code adapted from word similarity class

    Parameters
    ----------
    paragraph : string
        Content string from which features should be extracted.

    Returns
    -------
    dict : string -> float
    """
    AVG_SENTENCE_LENGTH = 15

    ##   Reading in locally cached Glove File
    global semantic_lexicon
    if semantic_lexicon == None:
      glv = utils.build_glove(os.path.join(path_to_glove, 
              'glove.6B.%dd.txt' % GLOVE_SIZE))
      semantic_lexicon = semantic_orientation_score_lexicon(mat=glv[0], rownames=glv[1],
        negset=('little', 'few', 'never', 'low', 'negative', 'wrong', 'small', 'inferior', 'seldom'),
        posset=('extremely', 'many', 'always', 'high', 'positive', 'correct', 'big', 'superior', 'often'))
      pickle.dump(semantic_lexicon, open( "data/semantic_lexicon.pickle", "w+" ) )

    ##   Calculate word and sentence sentiment variance
    intensity = []
    sentence = []
    sentence_means = []
    tokenized_and_lowercase = word_tokenize(paragraph.lower())
    punctuated = "." in tokenized_and_lowercase    #   Use this to account for punctuation
    #   punctuated = False    #   Use this when dataset has no punctuation
    for word in tokenized_and_lowercase:
        if word in semantic_lexicon:
            intensity.append(semantic_lexicon[word])
            sentence.append(semantic_lexicon[word])
        if (not punctuated and len(sentence) >= AVG_SENTENCE_LENGTH) \
        or (punctuated and word  in [".", "?", "!"]):
            m = np.mean(sentence)
            sentence_means.append(m)
            sentence = []
    m = np.mean(sentence)
    sentence_means.append(m)
    sentence_var = 0         # 0 variance when there's only 1 sentence
    if len(sentence_means) > 1:
        sentence_var = np.var(sentence_means)
    # print {'all_var': np.var(intensity), 'sentence_var': sentence_var}
    max_intensity = max(intensity)
    min_intensity = min(intensity)
    # return Counter({'all_var': np.var(intensity), 'sentence_var': sentence_var})
    return Counter({'min_max': max_intensity - min_intensity})

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
        - number of characters ==> 0.61 correlation [cur excluded]
        - number of white-space separated words (tokens) => 0.59
        - mean length of white-space separated tokens => 0.13
        - median length of white-space separated tokens [cur excluded]
        - count of words greater than length 6
    """
    MIN_LENGTH = 6
    result = Counter()
    tokens = word_tokenize(paragraph)

    result["length_in_characters"] = len(paragraph)
    result["length_in_words"] = len(tokens)

    word_lengths = [len(t) for t in tokens]
    result["length_mean_word_len"] = np.mean(word_lengths) #equiv to avg_word_length
    result["length_median_word_len"] = np.median(word_lengths) #equiv to median_frequency
    result["num_words_greater_than_6"] = len([w for w in tokens if len(w) > MIN_LENGTH])
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

def dimensional_decomposition(paragraph, unused_parse, num_dimensions_to_accumulate=10):
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

     Best to accumulate 10 dimensions -- we get as good performance as
     possible with 10 (no improvement with 20, 50, but some loss at 
     lower values). (This suggests human text complexities fit in 
     relatively low dimensional spaces -- but are more complex than that, 
     because using this feature alone doesn't lead to particularly high
     performance.)

     Can't use a large number of dimensions (like 50), else we risk
     a highly overdetermined system of equations when heading into PCA.

     Cross-validation on LogisticAT for ordinal regression prefers an 
     alpha of 0.2 with this problem formulation.

     This may not be capturing a lot beyond paragraph length, 
     unfortunately. Performance suggests that using only word length, we can
     get 0.33 +/- 0.06 correlation, using only dimensional, we get 0.39 +/- 0.03,
     and using both we're at 0.39 +/- 0.03 -- these are statistically noisy.
     So we unfortunately can't say much about whether PCA is getting more than 
     just number of words.

  Returns:
     dictionary, with keys:
        cum_pca_var_expl_# (where # is in range 
           [0, num_dimensions_to_accumulate)); each value gives the amount
           of variance explained when that number of dimensions 
           is considered
  """
  pca = derive_pca_on_glove(paragraph, num_dimensions_to_accumulate)

  features = Counter()
  for i in range(num_dimensions_to_accumulate):
    features["cum_pca_var_expl_%d" % i] = np.sum(pca.explained_variance_ratio_[:i+1])

  return features

def dimensional_decomposition_noncumulative(paragraph, unused_parse, 
  num_dimensions_to_accumulate=50):
  """ Similar to `dimensional_decomposition`, but does not cumsum the 
  eigenvalues -- instead returns the full set of `num_dimensions_to_accumulate`
  values as unique variables.  If the paragraph has fewer unique GloVe words
  than is `num_dimensions_to_accumulate`, pads resulting "variance explained
  by each dimension" vector with zero. 

  Non-cumulative performs worse than cumulative (e.g., correlations of 0.30
  instead of 0.39), so we opt against using it. 

  """
  
  pca = derive_pca_on_glove(paragraph, num_dimensions_to_accumulate)
  num_usable_dimensions = pca.explained_variance_ratio_.shape[0]

  features = Counter()
  for i in range(num_dimensions_to_accumulate):
    label = "explained_dim_"+str(i)
    if i < num_usable_dimensions:
      features[label] = pca.explained_variance_ratio_[i]
    else:
      features[label] = 0

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
  return Counter(features)

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
  np.seterr(all='raise')

  trees = []
  for t_string in parse:
    trees.append(Tree.fromstring(t_string))
  if len(trees) < 2:
    return Counter()
  first_tree = trees[0]
  first_sentence_tokens = utils_parsing.get_nouns_verbs([first_tree])
  rest_trees = trees[1:]
  rest_sentences_tokens = utils_parsing.get_nouns_verbs(rest_trees) 

  if len(rest_sentences_tokens) == 0:
    return Counter()

  similarities = np.zeros((len(first_sentence_tokens), len(rest_sentences_tokens)))
  for i, tuple1 in enumerate(first_sentence_tokens):
    for j, tuple2 in enumerate(rest_sentences_tokens):
      similarities[i,j] = utils_similarity.similarity_li(tuple1, tuple2)

  paragraph_semsim = np.exp(-np.sum(similarities.mean(axis=1)))

  return Counter({"kannan_ambili": paragraph_semsim})

############################
# Filtered length functions
############################

def keep_only(all_features, list_of_labels):
  features = Counter()
  for label in list_of_labels:
    features[label] = all_features[label]
  return features

def baseline_length(paragraph, unused_parse):
  """ Subsets the length features to only those used in the baseline """
  return keep_only(length(paragraph, unused_parse), ["length_in_words", "num_words_greater_than_6"])

def number_words_only(paragraph, unused_parse):
  """ Subsets the length features to only the number of words used """
  return keep_only(length(paragraph, unused_parse), ["length_in_words"])


#############################
# Helper functions
#############################
def wordlist_presence(wordlist_func, paragraph):
    """
    Helper. Produces feature list that is:
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
    """ Calculates the presence of modals """
    modals = wordlist_presence(utils_wordlists.get_modals, paragraph)
    tokens = word_tokenize(paragraph)
    
    modals["modal_count_token"] = np.sum( \
        [value for key, value in modals.items()])    
    modals["modal_count_type"] = len(modals) - 1 # -1 bc *_count_token
    modals["modal_freq"] = 1.0*modals["modal_count_token"] / len(tokens)
    
    return modals
    
def relative_amount_presence(paragraph, unused_parse):
    """ Calculates the presence of relative amount phrases """
    relamt = wordlist_presence(utils_wordlists.get_relative_amount, paragraph)
    relamt["relative_amount_count_token"] = np.sum( \
        [value for key, value in relamt.items()])    
    relamt["relative_amount_count_type"] = len(relamt) - 1 # -1 bc *_count_token
    return relamt

def transitional_presence(paragraph, unused_parse):
    """ Calculates the presence of transitional phrases """
    transitional = wordlist_presence(utils_wordlists.get_transitional, paragraph)
    transitional["transitional_count_token"] = np.sum( \
        [value for key, value in transitional.items()])    
    transitional["transitional_count_type"] = len(transitional) - 1 # -1 bc *_count_token
    return transitional
    
def hedge_presence(paragraph, unused_parse):
    """ Calculates the presence of hedges  """
    hedges = wordlist_presence(utils_wordlists.get_hedges, paragraph)
    hedges["hedge_count_token"] = np.sum( \
        [value for key, value in hedges.items()])    
    hedges["hedge_count_type"] = len(hedges) - 1 # -1 bc *_count_token
    return hedges

def conjunctives_presence(paragraph, unused_parse):
    """ Calculates the presence of conjunctives """
    conjunctives = wordlist_presence(utils_wordlists.get_conjunctives, paragraph)
    conjunctives["conjunctive_count_token"] = np.sum( \
        [value for key, value in conjunctives.items()])    
    conjunctives["conjunctive_count_type"] = len(conjunctives) - 1 # -1 bc *_count_token
    return conjunctives

def punctuation_presence(paragraph, unused_parse):
    """ Calculates the presence of punctuation """
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

def derive_pca_on_glove(paragraph, num_dimensions_to_accumulate):
  """ Helper function to get PCA decomposition of GLOVE vectors in paragraph """
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

  pca = PCA(n_components = num_dimensions_to_accumulate)
  pca.fit(word_vector)

  return pca



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
