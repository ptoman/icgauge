# -*- coding: utf-8 -*-
#!/usr/bin/python


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
        
        
def get_modals():
    """
    Returns a list of modal verbs in English.
    Note that "dare" and "need" are not always modal, but we include them
    regardless in case they are helpful.
    """
    return ["can", "could", "may", "might", "must", "shall", "should", 
            "will", "would", "ought", "had better", "dare", "need"]

def get_hedges():
    """
    Returns a list of hedge phrases in English, all lowercase. Tries to capture all 
    surface structure in terms of (a) verb forms, and (b) negation that appears
    inside the phrase itself, though better might be smarter addressing of these.
    GLOVE vectors might also be excellent to use to find similar words to this
    hedge set.
    """
    return ['"', 'I feel that', 'I think that', 'I would argue', 'a bit', 
            'a few', 'a lot', 'a lot of', 'about', 'aim', 'all i know', 
            'alleged', 'almost', 'almost always', 'almost never', 
            'am not sure', 'am sure', 'apparent', 'apparently', 
            'appear to', 'appearance', 'appeared to', 'appears to', 
            'approximately', 'are not sure', 'are sure', "aren't sure", 
            'arguable', 'arguably', 'argue', 'argued', 'argues', 'around', 
            'as a general rule', 'assume', 'assumed', 'assumes', 'assumption', 
            'ballpark', 'basically', 'be sure', 'believe', 'believed', 
            'believes', 'by the way', 'can', 'certain', 'certainly', 'claim', 
            'clear', 'clearly', 'comparatively', 'conceivably', 'conclusive', 
            'conclusively', 'confidently', 'consistent with', 'controversial', 
            'convincing', 'convincingly', 'could', 'debatable', 'definite', 
            'definitely', 'demonstrably', 'doubt', 'doubted', 'doubts', 
            'effectively', 'estimate', 'estimated', 'estimates', 'evidently', 
            'fairly', 'fairly often', 'feel', 'feels', 'felt', 'few', 
            'full consensus', 'generally', 'give or take', 'great', 
            'hardly ever', 'hopefully', 'hypothetically', 'if anything', 
            'if true', 'implied', 'implies', 'imply', 'in general', 
            'in part', 'in the area of', 'indicate', 'indicated', 'indicates', 
            'indication', 'infer', 'inference', 'intend', 'invalid', 
            'is not sure', 'is sure', "isn't sure", 'it can be the case that', 
            'it cannot be the case that', 'it could be the case that', 
            "it couldn't be the case that", 'it is important to', 
            'it is my view that', 'it is our view that', 'it is useful to', 
            'it may be possible to', 'it might be suggested that', 
            'it might not be possible', 'just', 'kind of', 'kinda', 
            'largely', 'likelihood', 'likely', 'look like', 'look probable', 
            'looked like', 'looks like', 'looks probable', 'mainly', 'may', 
            'maybe', 'might', 'more', 'more or less', 'most of the time', 
            'mostly', 'must', 'nearly', 'never', 'normally', 'not necessarily', 
            'occasionally', 'often', 'once in a while', 'or thereabouts', 
            'overall', 'partially', 'perhaps', 'possibility', 'possible', 
            'possibly', 'potentially', 'practically', 'predominantly', 
            'presumably', 'pretty', 'probability', 'probable', 'probably', 
            'propose', 'proposed', 'proposes', 'putative', 'quite', 
            'quite clearly', 'quite rarely', 'rarely', 'rather', 'really', 
            'really quite', 'reckon', 'reckoned', 'reckons', 'relatively', 
            'roughly', 'safely', 'seem', 'seem reasonable', 'seemed', 
            'seemingly', 'seems', 'seems reasonable', 'seen as', 'seldom', 
            'seldomly', 'several', 'shaky foundations', 'should', 'small', 
            'so to speak', 'some', 'somehow', 'sometimes', 'somewhat', 
            'sort of', 'speculate', 'speculated', 'speculates', 'suggest', 
            'suggested', 'suggestion', 'suggests', 'supposedly', 'tend', 
            'tended', 'tendency', 'tends', 'theoretically', 
            'there is every hope that', 'there is no doubt that', 
            'they told me that', 'they would argue', 'think', 'thinks', 
            'thought', 'to a certain degree', 'to generalise', 'to generalize', 
            'to my knowledge', 'to our knowledge', 'to some extent', 
            'typically', 'undeniable', 'undeniably', 'unfounded', 'unlikely', 
            'unreliable', 'usually', 'very', 'very often', 'virtually', 
            'virtually all', 'was not sure', 'was sure', "wasn't sure", 
            'we feel that', 'we think that', 'we would argue', 
            'well-documented', 'were not sure', 'were sure', "weren't sure", 
            'will', "won't", 'would']

def get_conjunctives():
    """
    Returns a list of conjunctive phrases in English, all lowercase. 
    Conjunctions, subordinating conjunctions, conjunctive adverbs, etc.
    """
    return ["'til", 'above all', 'accordingly', 'after', 'also', 'although', 
            'and', 'anyway', 'as', 'as a result', 'as if', 'as long as', 
            'as much as', 'as soon as', 'as though', 'as well', 
            'assuming that', 'because', 'before', 'besides', 'both', 'but', 
            'but also', 'by the time', 'consequently', 'conversely', 'either', 
            'even if', 'even though', 'eventually', 'ever since', 'except', 
            'except that', 'finally', 'firstly', 'for', 'furthermore', 
            'hardly', 'hence', 'how', 'however', 'if', 'in addition', 
            'in case', 'in order', 'in order that', 'indeed', 'instead', 
            'just as', 'just when', 'lest', 'like', 'likewise', 'meanwhile', 
            'moreover', 'neither', 'nevertheless', 'next', 'no sooner', 
            'nonetheless', 'nor', 'not only', 'now', 'now that', 
            'on the contrary', 'once', 'only if', 'or', 'otherwise', 
            'overall', 'provided that', 'rather', 'right before', 'scarcely', 
            'simply because', 'since', 'so', 'so as to', 'so that', 'still', 
            'than', 'that', 'then', 'therefore', 'though', 'thus', 'till', 
            'unless', 'until', 'what', 'what with', 'whatever', 'when', 
            'whenever', 'where', 'whereas', 'wherever', 'which', 'whichever', 
            'while', 'whilst', 'who', 'whoever', 'whom', 'whomever', 'whose', 
            'why', 'yet']
            
def get_punctuation():
    """
    Returns a list of punctuation.
    """
    return [".", ",", ";", "?", "!", "-", "'", "\"", ":", "(", ")", "[", "]"]
