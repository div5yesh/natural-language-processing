#%%
import os, sys
import numpy as np
path = "UD_English-EWT"
file_train = "en_ewt-ud-train.conllu"
file_dev = "en_ewt-ud-dev.conllu"

#%%
fp = open(os.path.join(path, file_train), "r", encoding='latin1')
data_train = fp.read()
fp.close()
print(len(data_train))

fp = open(os.path.join(path, file_dev), "r", encoding='latin1')
data_dev = fp.read()
fp.close()
print(len(data_dev))

#%%
temp = data_train.split("\n\n")
del temp[-1]
sentences_train = np.asarray(temp)
len_sentences_train = len(sentences_train)
print("Sentences:(train)", len_sentences_train)

temp = data_dev.split("\n\n")
del temp[-1]
sentences_dev = np.asarray(temp)
len_sentences_dev = len(sentences_dev)
print("Sentences:(dev)", len_sentences_dev)

#%%
def cleanup(sentence): 
    wrds = np.asarray(sentence.split("\n"))
    wrds = np.asarray(list(filter(lambda x: not x.startswith("#"), wrds)))
    return np.asarray(list(map(lambda x: x.split("\t"), wrds)))

wrds_train = np.asarray(list(map(cleanup, sentences_train)))
wrds_dev = np.asarray(list(map(cleanup, sentences_dev)))

#%%
wrd_avg_train = 0
for word in wrds_train:
    wrd_avg_train += len(word)

wrd_avg_train = wrd_avg_train/len_sentences_train
print("Avg words per sentence:(train)", wrd_avg_train)

wrd_avg_dev = 0
for word in wrds_dev:
    wrd_avg_dev += len(word)

wrd_avg_dev = wrd_avg_dev/len_sentences_dev
print("Avg words per sentence:(dev)", wrd_avg_dev)

#%%
types = []
for word in wrds_train:
    types += list(map(lambda x: x.lower(),word[:,1]))

unique_train, counts_train = np.unique(types, return_counts=True)
print("Tokens:(train)", len(types))
print("Types:(train)", len(unique_train))
types_train = dict(zip(unique_train, counts_train))

#%%
sorted_types_train = sorted(types_train.items(), key=lambda kv: kv[1], reverse=True)
print(sorted_types_train[0:50])

#%%
print(sorted_types_train[-50:])
print(list(filter(lambda x: x[1] == 10 ,sorted_types_train))[0:20])
print(list(filter(lambda x: x[1] == 20 ,sorted_types_train))[0:20])
print(list(filter(lambda x: x[1] == 50 ,sorted_types_train))[0:20])
print(list(filter(lambda x: x[1] == 101 ,sorted_types_train))[0:20])

#%%
types = []
for word in wrds_dev:
    types += list(map(lambda x: x.lower(),word[:,1]))

unique_dev, counts_dev = np.unique(types, return_counts=True)
print("Tokens:(dev)", len(types))
print("Types:(dev)", len(unique_dev))
types_dev = dict(zip(unique_dev, counts_dev))

#%%
oov = set(unique_dev) - set(unique_train)
print("OOV words:", len(oov))

#%%