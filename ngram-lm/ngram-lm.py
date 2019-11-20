#%%
import os, sys, math, pickle, argparse
import numpy as np

path_eng = "UD_English-EWT"
file_train_eng = "en_ewt-ud-train.conllu"
file_dev_eng = "en_ewt-ud-dev.conllu"
file_test_eng = "en_ewt-ud-test.conllu"

np.seterr(divide = 'ignore') 
#%%
class Corpus:
    def __init__(self, file):
        self.sentences = self.load_corpus(file)

    def load_corpus(self, file):
        fp = open(file, "r")
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
            sentence = ["<BOS>"] + list(word[:, 1]) + ["<EOS>"]
            tokens += list(word[:, 1])
            sentences.append(sentence)

        self.tokens = tokens
        self.vocab = set(np.unique(tokens))
        return np.asarray(sentences)

    def cleanup(self, sentence):
        wrds = sentence.split("\n")
        wrds = list(filter(lambda x: not x.startswith("#"), wrds))
        return np.asarray(list(map(lambda x: x.split("\t"), wrds)))


#%%
class NGramLM:
    def __init__(self, corpus, smoothing, backoff):
        self.corpus = corpus
        self.smoothing = smoothing
        self.backoff = backoff

    def getNGrams(self, N):
        NGramCounts = dict()
        for sentence in self.corpus.sentences:
            for idx in range(len(sentence)):
                key = tuple(sentence[idx:idx + N])
                NGramCounts[key] = NGramCounts.get(key, 0) + 1 + self.smoothing

        return NGramCounts

    def get1GramsProb(self, discount):
        N1GramCounts = self.getNGrams(1)
        Z = len(self.corpus.tokens) + self.smoothing * len(self.corpus.vocab)
        unigrams = dict(map(lambda kv: (kv[0], (kv[1] - discount)/ Z), N1GramCounts.items()))
        return unigrams

    def getNGramsProb(self, N, discount):
        NGramProb = dict()
        if N == 1:
            NGramProb = self.get1GramsProb(discount)
        else:
            NGramCounts = self.getNGrams(N)
            ZGramCounts = self.getNGrams(N - 1)
            for sentence in self.corpus.sentences:
                for idx in range(len(sentence)):
                    key = tuple(sentence[idx:idx + N])
                    history = tuple(sentence[idx:idx + N - 1])
                    Z = ZGramCounts[history] + self.smoothing * len(self.corpus.vocab)
                    NGramProb[key] = (NGramCounts[key] - discount) / Z

        return NGramProb

    def getBackoffProb(self, N, backoff):
        discount = backoff["discount"]
        threshold = backoff["threshold"]
        discountedunigramprob = self.get1GramsProb(discount)
        if N ==1:
            return discountedunigramprob

        bkprob = dict()
        bigramcounts = self.getNGrams(2)
        discountedbigrams = self.getNGramsProb(2, discount)
        bkoffweights = self.getBackoffWeight(bigramcounts)
        for sentence in self.corpus.sentences:
            for idx in range(len(sentence)):
                key = tuple(sentence[idx:idx + N])
                history = tuple(sentence[idx:idx + N - 1])
                if bigramcounts[key] > threshold:
                    bkprob[key] = discountedbigrams[key]
                else:
                    bkprob[key] = bkoffweights.get(history, 1) * discountedunigramprob[(key[1],)]

        return bkprob

    def getBackoffWeight(self, ngrams):
        weights = dict()
        for key in ngrams:
            if ngrams[key] == 0:
                weights[key[0]] = weights.get(key[0], 0) + 1

        Z = sum(weights.values())
        weights = dict(map(lambda kv: (kv[0], kv[1]/Z), weights.items()))
        return weights

    def save(self, data, file):
        pickle.dump(data, open(file, "wb"))

    def load(self, file):
        return pickle.load(open(file, "rb"))

    def getNGramPPL(self, N, test):
        log_prob = 0
        count = 0
        NGramProb = self.model
        for sentence in test.sentences:
            for idx in range(len(sentence)):
                key = tuple(sentence[idx:idx + N])
                if key in NGramProb and NGramProb > 0:
                    log_prob += np.log(NGramProb[key])
                    count += 1

        return np.exp(-log_prob / count)

    def train(self, N):
        model = self.getNGramsProb(N, 0)
        return {"model": model, "N": N}

    def trainBackoff(self, N):
        model = self.getBackoffProb(N, self.backoff)
        return {"model": model, "N": N}

    def eval(self, data):
        self.model = data["model"]
        ppl = self.getNGramPPL(data["N"], self.corpus)
        return ppl

# #%%
# # ------------------------------------------------------------debugging calls
# backoff = {"discount": 0.1, "threshold": 0}
# train = Corpus("UD_English-EWT/en_ewt-ud-train.conllu")
# lm2g_eng = NGramLM(train, 0, backoff)
# model = lm2g_eng.trainBackoff(2)
# lm2g_eng.save(model, "best-2gram_bkoff.lm")

# test = Corpus("UD_English-EWT/en_ewt-ud-dev.conllu")
# # test = Corpus("UD_English-EWT/en_ewt-ud-test.conllu")
# lm2g_eng = NGramLM(test, 0, backoff)
# model = lm2g_eng.load("best-2gram_bkoff.lm")
# print(lm2g_eng.eval(model))

# #%%
# train = Corpus("UD_English-EWT/en_ewt-ud-train.conllu")
# lm1g_eng = NGramLM(train, 100, None)
# model = lm1g_eng.train(1)
# lm1g_eng.save(model, "1gram.lm")

# #%%
# # test = Corpus("UD_English-EWT/en_ewt-ud-dev.conllu")
# test = Corpus("UD_English-EWT/en_ewt-ud-test.conllu")
# lm1g_eng = NGramLM(test, 0, None)
# # print(lm1g_eng.eval(model))
# model = lm1g_eng.load("1gram.lm")
# print(lm1g_eng.eval(model))

# #%%
# train = Corpus("UD_English-EWT/en_ewt-ud-train.conllu")
# lm2g_eng = NGramLM(train, 100, None)
# model = lm2g_eng.train(2)
# lm2g_eng.save(model, "2gram.lm")

# #%%
# # test = Corpus("UD_English-EWT/en_ewt-ud-dev.conllu")
# test = Corpus("UD_English-EWT/en_ewt-ud-test.conllu")
# lm2g_eng = NGramLM(test, 0, None)
# # print(lm2g_eng.eval(model))
# model = lm2g_eng.load("2gram.lm")
# print(lm2g_eng.eval(model))

# #%%
# train = Corpus("UD_English-EWT/en_ewt-ud-train.conllu")
# lm1g_eng = NGramLM(train, 1, None)
# model = lm1g_eng.train(1)
# lm1g_eng.save(model, "1gram.lm")

# test = Corpus("UD_English-EWT/en_ewt-ud-dev.conllu")
# lm1g_eng = NGramLM(test, 1, None)
# print(lm1g_eng.eval(model))
# model = lm1g_eng.load("1gram.lm")
# print(lm1g_eng.eval(model))

# #%%
# train = Corpus("UD_English-EWT/en_ewt-ud-train.conllu")
# lm2g_eng = NGramLM(train, 1, None)
# model = lm2g_eng.train(2)
# lm2g_eng.save(model, "2gram.lm")

# test = Corpus("UD_English-EWT/en_ewt-ud-dev.conllu")
# lm2g_eng = NGramLM(test, 1, None)
# print(lm2g_eng.eval(model))
# model = lm2g_eng.load("2gram.lm")
# print(lm2g_eng.eval(model))

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool)
    parser.add_argument('--corpus')
    parser.add_argument('--model')
    parser.add_argument('--lmtype')
    parser.add_argument('--tune')
    parser.add_argument('--hpargs')
    parser.add_argument('--N', type=int)
    args = parser.parse_args()
    if args.train:
        train = Corpus(args.corpus)
        lm_eng = NGramLM(train, 0, None)
        model = lm_eng.train(args.N)

        dev = Corpus(args.tune)
        lm_eng = NGramLM(dev, 0, args.hpargs)
        print(lm_eng.eval(model))
        lm_eng.save(model, args.model)
    else:
        test = Corpus(args.corpus)
        lm_eng = NGramLM(test, 0, None)
        model = lm_eng.load(args.model)
        print(lm_eng.eval(model))
