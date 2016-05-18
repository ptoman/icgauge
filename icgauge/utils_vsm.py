
# coding: utf-8

# # Distributional word representations

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2016 term"

import os
import sys
import csv
import random
import itertools
from operator import itemgetter
from collections import defaultdict
import numpy as np
import scipy
import scipy.spatial.distance
from numpy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
from sklearn.metrics import classification_report
import utils

def euclidean(u, v):    
    """Eculidean distance between 1d np.arrays `u` and `v`, which must 
    have the same dimensionality. Returns a float."""
    # Use scipy's method:
    return scipy.spatial.distance.euclidean(u, v)
    # Or define it yourself:
    # return vector_length(u - v)    

def vector_length(u):
    """Length (L2) of the 1d np.array `u`. Returns a new np.array with the 
    same dimensions as `u`."""
    return np.sqrt(np.dot(u, u))


def length_norm(u):
    """L2 norm of the 1d np.array `u`. Returns a float."""
    return u / vector_length(u)

def cosine(u, v):        
    """Cosine distance between 1d np.arrays `u` and `v`, which must have 
    the same dimensionality. Returns a float."""
    # Use scipy's method:
    return scipy.spatial.distance.cosine(u, v)
    # Or define it yourself:
    # return 1.0 - (np.dot(u, v) / (vector_length(u) * vector_length(v)))

def matching(u, v):    
    """Matching coefficient between the 1d np.array vectors `u` and `v`, 
    which must have the same dimensionality. Returns a float."""
    # The scipy implementation is for binary vectors only. 
    # This version is more general.
    return np.sum(np.minimum(u, v))

def jaccard(u, v):    
    """Jaccard distance between the 1d np.arrays `u` and `v`, which must 
    have the same dimensionality. Returns a float."""
    # The scipy implementation is for binary vectors only. 
    # This version is more general.
    return 1.0 - (matching(u, v) / np.sum(np.maximum(u, v)))

def neighbors(word, mat, rownames, distfunc=cosine):    
    """Tool for finding the nearest neighbors of `word` in `mat` according 
    to `distfunc`. The comparisons are between row vectors.
    
    Parameters
    ----------
    word : str
        The anchor word. Assumed to be in `rownames`.
        
    mat : np.array
        The vector-space model.
        
    rownames : list of str
        The rownames of mat.
            
    distfunc : function mapping vector pairs to floats (default: `cosine`)
        The measure of distance between vectors. Can also be `euclidean`, 
        `matching`, `jaccard`, as well as any other distance measure  
        between 1d vectors.
        
    Raises
    ------
    ValueError
        If word is not in rownames.
    
    Returns
    -------    
    list of tuples
        The list is ordered by closeness to `word`. Each member is a pair 
        (word, distance) where word is a str and distance is a float.
    
    """
    if word not in rownames:
        raise ValueError('%s is not in this VSM' % word)
    w = mat[rownames.index(word)]
    dists = [(rownames[i], distfunc(w, mat[i])) for i in range(len(mat))]
    return sorted(dists, key=itemgetter(1), reverse=False)

def prob_norm(u):
    """Normalize 1d np.array `u` into a probability distribution. Assumes 
    that all the members of `u` are positive. Returns a 1d np.array of 
    the same dimensionality as `u`."""
    return u / np.sum(u)

def pmi(mat, rownames=None, positive=True):
    """Pointwise Mutual Information with Positive on by default.
    
    Parameters
    ----------
    mat : 2d np.array
       The matrix to operate on.
           
    rownames : list of str or None
        Not used; it's an argument only for consistency with other methods 
        defined here.
        
    positive : bool (default: True)
        Implements Positive PMI.
        
    Returns
    -------    
    (np.array, list of str)
       The first member is the PMI-transformed version of `mat`, and the 
       second member is `rownames` (unchanged).
    
    """    
    # Joint probability table:
    p = mat / np.sum(mat, axis=None)
    # Pre-compute column sums:
    colprobs = np.sum(p, axis=0)
    # Vectorize this function so that it can be applied rowwise:
    np_pmi_log = np.vectorize((lambda x : _pmi_log(x, positive=positive)))
    p = np.array([np_pmi_log(row / (np.sum(row)*colprobs)) for row in p])   
    return (p, rownames)

def _pmi_log(x, positive=True):
    """With `positive=False`, return log(x) if possible, else 0.
    With `positive=True`, log(x) is mapped to 0 where negative."""
    val = 0.0
    if x > 0.0:
        val = np.log(x)
    if positive:
        val = max([val,0.0])
    return val

def tfidf(mat, rownames=None):
    """TF-IDF 
    
    Parameters
    ----------
    mat : 2d np.array
       The matrix to operate on.
       
    rownames : list of str or None
        Not used; it's an argument only for consistency with other methods 
        defined here.
        
    Returns
    -------
    (np.array, list of str)    
       The first member is the TF-IDF-transformed version of `mat`, and 
       the second member is `rownames` (unchanged).
    
    """
    colsums = np.sum(mat, axis=0)
    doccount = mat.shape[1]
    w = np.array([_tfidf_row_func(row, colsums, doccount) for row in mat])
    return (w, rownames)

def _tfidf_row_func(row, colsums, doccount):
    df = float(len([x for x in row if x > 0]))
    idf = 0.0
    # This ensures a defined IDF value >= 0.0:
    if df > 0.0 and df != doccount:
        idf = np.log(doccount / df)
    tfs = row/colsums
    return tfs * idf


def lsa(mat=None, rownames=None, k=100):
    """Latent Semantic Analysis using pure scipy.
    
    Parameters
    ----------
    mat : 2d np.array
       The matrix to operate on.
           
    rownames : list of str or None
        Not used; it's an argument only for consistency with other methods 
        defined here.
        
    k : int (default: 100)
        Number of dimensions to truncate to.
        
    Returns
    -------    
    (np.array, list of str)
        The first member is the SVD-reduced version of `mat` with 
        dimension (m x k), where m is the rowcount of mat and `k` is 
        either the user-supplied k or the column count of `mat`, whichever 
        is smaller. The second member is `rownames` (unchanged).

    """    
    rowmat, singvals, colmat = svd(mat, full_matrices=False)
    singvals = np.diag(singvals)
    trunc = np.dot(rowmat[:, 0:k], singvals[0:k, 0:k])
    return (trunc, rownames)


def glove(mat, rownames, n=100, xmax=100, alpha=0.75, 
          iterations=100, learning_rate=0.05, 
          display_progress=True):
    """Basic GloVe. 
    
    Parameters
    ----------
    mat : 2d np.array
        This must be a square count matrix.
        
    rownames : list of str or None
        Not used; it's an argument only for consistency with other methods 
        defined here.
        
    n : int (default: 100)
        The dimensionality of the output vectors.
    
    xmax : int (default: 100)
        Words with frequency greater than this are given weight 1.0.
        Words with frequency under this are given weight (c/xmax)**alpha
        where c is their count in mat (see the paper, eq. (9)).
        
    alpha : float (default: 0.75)
        Exponent in the weighting function (see the paper, eq. (9)).
    
    iterations : int (default: 100)
        Number of training epochs.
        
    learning_rate : float (default: 0.05)
        Controls the rate of SGD weight updates.
        
    display_progress : bool (default: True) 
        Whether to print iteration number and current error to stdout.
        
    Returns
    -------
    (np.array, list of str)
       The first member is the learned GloVe matrix and the second is
       `rownames` (unchanged).

    """        
    m = mat.shape[0]
    W = utils.randmatrix(m, n) # Word weights.
    C = utils.randmatrix(m, n) # Context weights.
    B = utils.randmatrix(2, m) # Word and context biases.
    indices = list(range(m))
    for iteration in range(iterations):
        error = 0.0        
        random.shuffle(indices)
        for i, j in itertools.product(indices, indices):
            if mat[i,j] > 0.0:     
                # Weighting function from eq. (9)
                weight = (mat[i,j] / xmax)**alpha if mat[i,j] < xmax else 1.0
                # Cost is J' based on eq. (8) in the paper:
                diff = np.dot(W[i], C[j]) + B[0,i] + B[1,j] - np.log(mat[i,j])                
                fdiff = diff * weight                
                # Gradients:
                wgrad = fdiff * C[j]
                cgrad = fdiff * W[i]
                wbgrad = fdiff
                wcgrad = fdiff
                # Updates:
                W[i] -= (learning_rate * wgrad) 
                C[j] -= (learning_rate * cgrad) 
                B[0,i] -= (learning_rate * wbgrad) 
                B[1,j] -= (learning_rate * wcgrad)                 
                # One-half squared error term:                              
                error += 0.5 * weight * (diff**2)
        if display_progress:
            utils.progress_bar("iteration %s: error %s" % (iteration, error))
    if display_progress:
        sys.stderr.write('\n')
    # Return the sum of the word and context matrices, per the advice 
    # in section 4.2:
    return (W + C, rownames)


def semantic_orientation(
        mat, 
        rownames,
        seeds1=('bad', 'nasty', 'poor', 'negative', 'unfortunate', 'wrong', 'inferior'),
        seeds2=('good', 'nice', 'excellent', 'positive', 'fortunate', 'correct', 'superior'),
        distfunc=cosine):    
    """No frills implementation of the semantic Orientation (SO) method of 
    Turney and Littman. seeds1 and seeds2 should be representative members 
    of two intutively opposing semantic classes. The method will then try 
    to rank the vocabulary by its relative association with each seed set.
        
    Parameters
    ----------
    mat : 2d np.array
        The matrix used to derive the SO ranking.
        
    rownames : list of str
        The names of the rows of `mat` (the vocabulary).
        
    seeds1 : tuple of str
        The default is the negative seed set of Turney and Littman.
        
    seeds2 : tuple of str
        The default is the positive seed set of Turney and Littman.
        
    distfunc : function mapping vector pairs to floats (default: `cosine`)
        The measure of distance between vectors. Can also be `euclidean`, 
        `matching`, `jaccard`, as well as any other distance measure 
        between 1d vectors. 
    
    Returns
    -------    
    list of tuples
        The vocabulary ranked according to the SO method, with words 
        closest to `seeds1` at the top and words closest to `seeds2` at the 
        bottom. Each member of the list is a (word, score) pair.
    
    """    
    sm1 = _so_seed_matrix(seeds1, mat, rownames)
    sm2 = _so_seed_matrix(seeds2, mat, rownames)
    scores = [(rownames[i], _so_row_func(mat[i], sm1, sm2, distfunc)) for i in range(len(mat))]
    return sorted(scores, key=itemgetter(1), reverse=False)

def _so_seed_matrix(seeds, mat, rownames):
    indices = [rownames.index(word) for word in seeds if word in rownames]
    if not indices:
        raise ValueError('The matrix contains no members of the seed set: %s' % ",".join(seeds))
    return mat[np.array(indices)]
    
def _so_row_func(row, sm1, sm2, distfunc):
    val1 = np.sum([distfunc(row, srow) for srow in sm1])
    val2 = np.sum([distfunc(row, srow) for srow in sm2])
    return val1 - val2    


def wordsim_dataset_reader(src_filename, header=False, delimiter=','):    
    """Basic reader that works for all four files, since they all have the 
    format word1,word2,score, differing only in whether or not they include 
    a header line and what delimiter they use.
    
    Parameters
    ----------
    src_filename : str
        Full path to the source file.
        
    header : bool (default: False)
        Whether `src_filename` has a header.
        
    delimiter : str (default: ',')
        Field delimiter in `src_filename`.
    
    Yields
    ------    
    (str, str, float)
       (w1, w2, score) where `score` is the negative of the similarity 
       score in the file so that we are intuitively aligned with our 
       distance-based code.
    
    """
    reader = csv.reader(open(src_filename), delimiter=delimiter)
    if header:
        next(reader)
    for row in reader:
        w1, w2, score = row
        # Negative of scores to align intuitively with distance functions:
        score = -float(score)
        yield (w1, w2, score)

def wordsim353_reader():
    """WordSim-353: http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/"""
    src_filename = os.path.join(vsmdata_home, 'wordsim', 'wordsim353.csv')
    return wordsim_dataset_reader(src_filename, header=True)
 
def mturk287_reader():
    """MTurk-287: http://tx.technion.ac.il/~kirar/Datasets.html"""
    src_filename = os.path.join(vsmdata_home, 'wordsim', 'MTurk-287.csv')
    return wordsim_dataset_reader(src_filename, header=False)
    
def mturk771_reader():
    """MTURK-771: http://www2.mta.ac.il/~gideon/mturk771.html"""
    src_filename = os.path.join(vsmdata_home, 'wordsim', 'MTURK-771.csv')
    return wordsim_dataset_reader(src_filename, header=False)

def men_reader():
    """MEN: http://clic.cimec.unitn.it/~elia.bruni/MEN"""
    src_filename = os.path.join(vsmdata_home, 'wordsim', 'MEN_dataset_natural_form_full')
    return wordsim_dataset_reader(src_filename, header=False, delimiter=' ')    


def word_similarity_evaluation(reader, mat, rownames, distfunc=cosine):
    """Word-similarity evalution framework.
    
    Parameters
    ----------
    reader : iterator
        A reader for a word-similarity dataset. Just has to yield
        tuples (word1, word2, score).
    
    mat : 2d np.array
        The VSM being evaluated.
        
    rownames : list of str
        The names of the rows in mat.
        
    distfunc : function mapping vector pairs to floats (default: `cosine`)
        The measure of distance between vectors. Can also be `euclidean`, 
        `matching`, `jaccard`, as well as any other distance measure 
        between 1d vectors.  
    
    Prints
    ------
    To standard output
        Size of the vocabulary overlap between the evaluation set and
        rownames. We limit the evalation to the overlap, paying no price
        for missing words (which is not fair, but it's reasonable given
        that we're working with very small VSMs in this notebook).
    
    Returns
    -------
    float
        The Spearman rank correlation coefficient between the dataset
        scores and the similarity values obtained from `mat` using 
        `distfunc`. This evaluation is sensitive only to rankings, not
        to absolute values.
    
    """    
    sims = defaultdict(list)
    vocab = set([])
    for w1, w2, score in reader():
        if w1 in rownames and w2 in rownames:
            sims[w1].append((w2, score))
            sims[w2].append((w1, score))
            vocab.add(w1)
            vocab.add(w2)
    print("Evaluation vocabulary size: %s" % len(vocab))
    # Evaluate the matrix by creating a vector of all_scores for data
    # and all_dists for mat's distances. 
    all_scores = []
    all_dists = []
    for word in vocab:
        vec = mat[rownames.index(word)]
        vals = sims[word]
        cmps, scores = zip(*vals)
        all_scores += scores
        all_dists += [distfunc(vec, mat[rownames.index(w)]) for w in cmps]
    # Return just the rank correlation coefficient (index [1] would be the p-value):
    return scipy.stats.spearmanr(all_scores, all_dists)[0]   

def full_word_similarity_evaluation(mat, rownames):
    """Evaluate the (mat, rownames) VSM against all four datasets."""
    for reader in (wordsim353_reader, mturk287_reader, mturk771_reader, men_reader):
        print("="*40)
        print(reader.__name__)
        print('Spearman r: %0.03f' % word_similarity_evaluation(reader, mat, rownames))

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log2(a/b), 0))
    
def jensen_shannon(u, v):
    """Jensen Shannon distance between the 1d np.arrays `u` and `v`, 
    is a symmetrized and smoothed version of the Kullback–Leibler divergenc.
    Jensen Shannon distance equals the average distance between `u` and `v`
    and their middle ground."""
    """The method normalizes both vectors before computations"""
    u = prob_norm(u)
    v = prob_norm(v)
    m = 0.5 * u + 0.5 * v
    return 0.5 * KL(u, m) + 0.5 * KL(v, m)


def dice_coefficient(u, v):
    """Computation of dice coefficient of vector u and v."""
    u = np.asarray(u, dtype=np.float)
    v = np.asarray(v, dtype=np.float)

    # Compute Dice coefficient
    intersection = 2 * np.sum(np.minimum(u, v))/ np.sum(u + v)
    return 1. - intersection

def ttest(mat, rownames=None):
    """t-test reweighting
    
    Parameters
    ----------
    mat : 2d np.array
       The matrix to operate on.
       
    rownames : list of str or None
        Not used; it's an argument only for consistency with other methods 
        defined here.
        
    Returns
    -------
    (np.array, list of str)    
       The first member is the t-test-transformed version of `mat`, and 
       the second member is `rownames` (unchanged).
    
    """
    # Joint probability table:
    p = mat / np.sum(mat, axis=None)
    (nrow, ncol) = mat.shape
    # Pre-compute column sums:
    colprobs = np.sum(p, axis=0).reshape(1, ncol)
    # Pre-compute row sums:
    rowprobs = np.sum(p, axis=1).reshape(nrow, 1)
    
    intersection = rowprobs * colprobs
    p = np.multiply((p - intersection), 1./np.sqrt(intersection))
    return (p, rownames)

def semantic_orientation_lexicon(mat, rownames, posset, negset, distfunc=cosine):
    """Induce a pos/neg lexicon using `semantic_orientation`.
    
    Parameters
    ----------
    mat : 2d np.array
        The matrix to use.
        
    rownames : list
        The vocabulary for the rows in `mat`.
        
    posset : list of str
        Positive seed-set.

    negset : list of str
        Negative seed-set
        
    distfunc : (default: `cosine`)
        Distance function on vectors.
    
    Returns
    -------
    dict
        A dict with keys 'positive' and 'negative', where the 
        values are subsets of `rownames` deemed to be positive
        and negative, respectively.
    
    """
    # Use `semantic_orientation` to get a ranking of the full vocabulary:
    ranking = semantic_orientation(mat=mat, rownames=rownames, seeds1= negset, seeds2=posset)
        
    # This will split `ranking` into the vocabulary and vals lists.    
    words, vals = zip(*ranking)
    
    # You can then do some work with `vals` to find a boundary or 
    # boundaries for decision making. There are lots of ways to do this.
        
    # Use your decision-making procedure to derive from `words` a list of 
    # positive words and a list of negative words.
    poswords = []
    negwords = []
    # Use median value as threshold
    threshold = np.median(vals)
    for i in range(len(words)):
        if (vals[i] >= threshold):
            poswords.append(words[i])
        else:
            negwords.append(words[i])
    return {'positive': poswords, 'negative': negwords}

def evaluate_semantic_orientation_lexicon(lexicon):
    """Evaluates `lexicon`, which is the output of `semantic_orientation_lexicon`."""    
    # Read in the assessment lexicon:
    assessment_lexicon = pickle.load(file(os.path.join(vsmdata_home, 'imdb-posneg-lexicon.pickle')))
    # Full vocab for evaluation:
    vocab = assessment_lexicon['positive_train'] + assessment_lexicon['negative_train']
    # Gold data:
    gold = ['positive' for w in assessment_lexicon['positive_train']]
    gold += ['negative' for w in assessment_lexicon['negative_train']]
    
    # Use `lexicon` to get a list of predictions:            
    predictions = None # To be replaced.
    
    # Use `lexicon` to get a list of predictions:
    predictions = []
    for w in assessment_lexicon['positive_train']:
        if w in lexicon['negative']:
            predictions.append('negative')
        else:
            predictions.append('positive')
    for w in assessment_lexicon['negative_train']:
        if w in lexicon['negative']:
            predictions.append('negative')
        else:
            predictions.append('positive')

    # Finally, return the value of `sklearn.metrics.classification_report`
    # assessing gold vs. predictions:
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html    
    return classification_report(gold, predictions)

"""
evaluate_semantic_orientation_lexicon(semantic_orientation_lexicon(mat=ww_ppmi[0], rownames=ww_ppmi[1],
        negset=('bad', 'nasty', 'poor', 'negative', 'unfortunate', 'wrong', 'inferior', 'low'),
        posset=('good', 'nice', 'excellent', 'positive', 'fortunate', 'correct', 'superior', 'high')))
"""

def semantic_orientation_score_lexicon(mat, rownames, posset, negset, distfunc=cosine):
    """Induce a pos/neg lexicon using `semantic_orientation`.
    
    Parameters
    ----------
    mat : 2d np.array
        The matrix to use.
        
    rownames : list
        The vocabulary for the rows in `mat`.
        
    posset : list of str
        Positive seed-set.

    negset : list of str
        Negative seed-set
        
    distfunc : (default: `cosine`)
        Distance function on vectors.
    
    Returns
    -------
    dict
        A dict with keys 'positive' and 'negative', where the 
        values are subsets of `rownames` deemed to be positive
        and negative, respectively.
    
    """
    # Use `semantic_orientation` to get a ranking of the full vocabulary:
    ranking = semantic_orientation(mat=mat, rownames=rownames, seeds1= negset, seeds2=posset)
        
    # This will split `ranking` into the vocabulary and vals lists.    
    words, vals = zip(*ranking)
    
    # You can then do some work with `vals` to find a boundary or 
    # boundaries for decision making. There are lots of ways to do this.
        
    # Use your decision-making procedure to derive from `words` a list of 
    # positive words and a list of negative words.
    poswords = []
    negwords = []
    # Use median value as threshold
    threshold = np.median(vals)
    result = {}
    for i in range(len(words)):
        result[words[i]] = vals[i]
        
    return result