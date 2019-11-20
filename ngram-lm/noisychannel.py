#%%
import os, sys, math, pickle, argparse
import numpy as np
import sklearn.metrics as metrics

path_eng1 = "UD_English-EWT"
file_train_eng1 = "en_ewt-ud-train.conllu"
file_dev_eng1 = "en_ewt-ud-dev.conllu"
file_test_eng1 = "en_ewt-ud-test.conllu"

# path_eng2 = "UD_English-GUM"
# file_train_eng2 = "en_gum-ud-train.conllu"
# file_dev_eng2 = "en_gum-ud-dev.conllu"
# file_test_eng2 = "en_gum-ud-test.conllu"

path_eng2 = "UD_French-GSD"
file_train_eng2 = "fr_gsd-ud-train.conllu"
file_dev_eng2 = "fr_gsd-ud-dev.conllu"
file_test_eng2 = "fr_gsd-ud-test.conllu"

# path_eng3 = "UD_English-ParTUT"
# file_train_eng3 = "en_partut-ud-train.conllu"
# file_dev_eng3 = "en_partut-ud-dev.conllu"
# file_test_eng3 = "en_partut-ud-test.conllu"

path_eng3 = "UD_Hindi-HDTB"
file_train_eng3 = "hi_hdtb-ud-train.conllu"
file_dev_eng3 = "hi_hdtb-ud-dev.conllu"
file_test_eng3 = "hi_hdtb-ud-test.conllu"

#%%
class Corpus:
    def __init__(self, file):
        self.sentences = self.load_corpus(file)

    def load_corpus(self, file):
        fp = open(file, "r", encoding='utf-8')
        data = fp.read()
        fp.close()

        corpus = data.split("\n\n")
        del corpus[-1]

        sentences = self.getSentences(corpus)
        return sentences

    def getSentences(self, corpus):
        sentences = []
        tokens = []
        words = list(map(self.cleanup, corpus))
        for word in words:
            sentence = list(word[:, 1])
            tokens += list(word[:, 1])
            sentences.append(sentence)

        self.tokens = tokens
        self.vocab = np.unique(tokens)
        return np.asarray(sentences)

    def cleanup(self, sentence):
        wrds = sentence.split("\n")
        wrds = list(filter(lambda x: not x.startswith("#"), wrds))
        return np.asarray(list(map(lambda x: x.split("\t"), wrds)))

#%%
class NoisyChannel:
    def __init__(self, corpora):
        self.l = len(corpora)
        self.corpora = corpora
        self.prior = self.getPrior()

    def getPrior(self):
        prior = dict()
        for i in range(self.l):
            prior[i] = len(self.corpora[i].sentences)

        Z = sum(prior.values())
        return dict(map(lambda kv: (kv[0], kv[1]/Z), prior.items()))

    def getLikelihood(self, l, corpus):
        dist = dict()
        for sentence in corpus.sentences:
            for word in sentence:
                key = (l, word)
                dist[key] = dist.get(key, 0) + 1

        Z = len(corpus.tokens)
        pwl = dict(map(lambda kv:(kv[0],kv[1]/Z), dist.items()))
        return pwl

    def eval_baseline(self, corpora):
        actual = []
        sentences = 0
        for idx, corpus in enumerate(corpora):
            for sentence in corpora[corpus].sentences:
                actual.append(idx)
                sentences+=1

        pred = np.full(sentences, np.argmax(self.prior))
        self.scores(actual, pred)

    def train(self):
        pwl = dict()
        for i in range(self.l):
            pwl.update(self.getLikelihood(i, self.corpora[i]))

        self.likelihood = pwl

    def eval(self, corpora):
        pred = []
        actual = []
        for idx, corpus in enumerate(corpora):
            for sentence in corpora[corpus].sentences:
                actual.append(idx)
                posterior = np.zeros(self.l)
                for i in range(self.l):
                    likelihood = 0
                    for word in sentence:
                        key = (i, word)
                        likelihood += math.log(self.likelihood.get(key,1))
                    likelihood += math.log(self.prior[i])
                    posterior[i] = abs(likelihood)
                pred.append(np.argmax(posterior))
        
        self.scores(actual, pred)

    def scores(self, actual, pred):
        print("Accuracy:", metrics.accuracy_score(actual, pred))
        print(metrics.classification_report(actual, pred))

#%%
corpora = dict()
corpus1 = Corpus(path_eng1+"/"+file_train_eng1)
corpora[0] = corpus1

corpus2 = Corpus(path_eng2+"/"+file_train_eng2)
corpora[1] = corpus2

corpus3 = Corpus(path_eng3+"/"+file_train_eng3)
corpora[2] = corpus3

corpora_dev = dict()
corpus1 = Corpus(path_eng1+"/"+file_dev_eng1)
corpora_dev[0] = corpus1

corpus2 = Corpus(path_eng2+"/"+file_dev_eng2)
corpora_dev[1] = corpus2

corpus3 = Corpus(path_eng3+"/"+file_dev_eng3)
corpora_dev[2] = corpus3

corpora_test = dict()
corpus1 = Corpus(path_eng1+"/"+file_test_eng1)
corpora_test[0] = corpus1

corpus2 = Corpus(path_eng2+"/"+file_test_eng2)
corpora_test[1] = corpus2

corpus3 = Corpus(path_eng3+"/"+file_test_eng3)
corpora_test[2] = corpus3

#%%
model = NoisyChannel(corpora)
baseline = np.argmax(model.prior)
model.train()

#%%
# Dev
model.eval_baseline(corpora_dev)
model.eval(corpora_dev)

#%%
# Test
model.eval(corpora_test)

#%%
