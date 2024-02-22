import nltk
import random

# extract kw from given stc
def keyword_extractor(cap_stc, count):
    # low every alph
    cap_stc = cap_stc.lower()
    # noun, verb, adj will be keys (exclude stopwords and puncs)
    is_noun = lambda pos: pos[:2] == 'NN'
    is_verb = lambda pos: pos[:2] == 'VB'
    is_adj = lambda pos: pos[:2] == 'JJ'
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '<|endoftext|>', 'another']
    stop_words = nltk.corpus.stopwords.words("english")
    cap_words = nltk.word_tokenize(cap_stc)
    cap_kws = [word for (word, pos) in nltk.pos_tag(cap_words) if ((word not in english_punctuations) and (word not in stop_words) and (is_noun or is_verb or is_adj))]
    if len(cap_kws) > 8:
        cap_kws = random.sample(cap_kws, 5)
    min_keys = max(0, len(cap_kws) - (count//5))
    num_keys = max(0, len(cap_kws))
    num_keys = random.randint(min_keys, num_keys)
    cap_kws = random.sample(cap_kws, num_keys)
    # random.shuffle(cap_kws)
    if not cap_kws:
        return ','
    kw = ''
    for key in cap_kws:
        kw = kw + key + ','
    # using comma between kws
    return kw