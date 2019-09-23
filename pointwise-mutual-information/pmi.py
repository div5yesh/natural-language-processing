#%%
import os, sys, math
import numpy as np
import matplotlib.pyplot as plt

path_eng = "UD_English-EWT"
file_train_eng = "en_ewt-ud-train.conllu"

#%%
def getSentencesFromLanguage(path, file):
    fp = open(os.path.join(path, file), "r", encoding='utf-8')
    data = fp.read()
    fp.close()

    temp = data.split("\n\n")
    del temp[-1]
    return np.asarray(temp)

#%%
def cleanup(sentence): 
    wrds = np.asarray(sentence.split("\n"))
    wrds = np.asarray(list(filter(lambda x: not x.startswith("#"), wrds)))
    return np.asarray(list(map(lambda x: x.split("\t"), wrds)))

#%%
def getTokenFreqDistr(corpus):
    tokens = []
    sentences = []

    words = np.asarray(list(map(cleanup, corpus)))
    for word in words:
        sentences.append(word[:,1])
        tokens += list(word[:,1]) #list(map(lambda x: x.lower(),word[:,1]))

    types, counts = np.unique(tokens, return_counts=True)
    distribution = dict(zip(types, counts))
    return [sentences, distribution]

#%%
corpus = getSentencesFromLanguage(path_eng, file_train_eng)
sentences, distribution = getTokenFreqDistr(corpus)

#%%
sorted_distribution = np.asarray(sorted(distribution.items(), key=lambda kv: kv[1], reverse=True))
print("Types:", len(distribution)) #19672
print("Sentences: ", len(corpus)) #sentences=12543
print("g(cat):", distribution['cat']) #42

#%%
def getBigrams(sentences, param=0):
    bigrams = dict()
    bigrams_marginal = dict()
    for sentence in sentences:
        len_sentence = len(sentence)
        for idx1 in range(len_sentence):
            for idx2 in range(idx1 + 1, len_sentence):
                wrd1 = sentence[idx1]
                wrd2 = sentence[idx2]
                if wrd1 in sentence and wrd2 in sentence and wrd1 != wrd2:
                    key = (wrd1,wrd2)
                    if key in bigrams:
                        bigrams[key] += 1
                    else:
                        bigrams[key] = 1 + param

                    if wrd1 in bigrams_marginal:
                        bigrams_marginal[wrd1] += 1
                    else:
                        bigrams_marginal[wrd1] = 1 + param

                    if wrd2 in bigrams_marginal:
                        bigrams_marginal[wrd2] += 1
                    else:
                        bigrams_marginal[wrd2] = 1 + param
    return [bigrams, bigrams_marginal]
                
#%%
def getAssociations(term, p_marginal, p_bigrams):
    associations = dict()
    p_bigrams_filtered = dict(filter(lambda kv: term in kv[0], p_bigrams.items()))
    for bigram in p_bigrams_filtered:
        p_wrd1 = p_marginal[bigram[0]]
        p_wrd2 = p_marginal[bigram[1]]
        ratio = p_bigrams_filtered[bigram]/(p_wrd1 * p_wrd2)
        associations[bigram] = math.log2(ratio)
    return associations

#%%
def getLambdaAssociations(sentences, param=0):
    bigrams, bigrams_marginal = getBigrams(sentences, param)
    normalization = sum(bigrams.values())
    print("Normalization(Z):", normalization)

    p_marginal = dict(map(lambda kv: (kv[0], kv[1]/normalization), bigrams_marginal.items()))
    p_bigrams = dict(map(lambda kv: (kv[0], kv[1]/normalization), bigrams.items()))

    p_marginal_sorted = list(sorted(p_marginal.items(), key=lambda kv: kv[1], reverse=True))
    p_bigrams_sorted = list(sorted(p_bigrams.items(), key=lambda kv: kv[1], reverse=True))
    print("Top marginals:", p_marginal_sorted[:10])
    print("Top bigrams:", p_bigrams_sorted[:10])

    associations = getAssociations("doctor", p_marginal, p_bigrams)
    associations_sorted = list(sorted(associations.items(), key=lambda kv: kv[1], reverse=True))
    print("Top doctor associations:", associations_sorted[:10])
    print("Bottom doctor associations:", associations_sorted[-10:])

#%%
# L=0
getLambdaAssociations(sentences)

#%%
# L=[0.003,0.2,1,2,10]
getLambdaAssociations(sentences, 0.2)

#%%
