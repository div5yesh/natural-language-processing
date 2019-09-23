#%%
import os, sys, math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')

path_eng = "UD_English-EWT"
path_hin = "UD_Hindi-HDTB"
file_train_eng = "en_ewt-ud-train.conllu"
file_train_hin = "hi_hdtb-ud-train.conllu"

#%%
plt.rcParams['axes.edgecolor']='black'
plt.rcParams['xtick.color']='black'
plt.rcParams['ytick.color']='black'
plt.rcParams['axes.facecolor']='white'
plt.rcParams['axes.labelcolor']='black'
plt.rcParams['text.color']='black'
plt.rcParams['figure.facecolor']='white'

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
def getTokenFreqDistr(sentences):
    types = []
    words = np.asarray(list(map(cleanup, sentences)))
    for word in words:
        types += list(map(lambda x: x.lower(),word[:,1]))

    unique, counts = np.unique(types, return_counts=True)
    # freq_rank = list(zip(counts, 1/counts))
    distribution = list(zip(unique, np.log10(counts), counts))
    return distribution

#%%
sentences_eng = getSentencesFromLanguage(path_eng, file_train_eng)
sentences_hin = getSentencesFromLanguage(path_hin, file_train_hin)

#%%
dist_eng = getTokenFreqDistr(sentences_eng)
sorted_dist_eng = np.asarray(sorted(dist_eng, key=lambda kv: kv[1], reverse=True))
dist_hin = getTokenFreqDistr(sentences_hin)
sorted_dist_hin = np.asarray(sorted(dist_hin, key=lambda kv: kv[1], reverse=True))

#%%
x_eng = np.asarray([math.log10(i+1) for i in range(len(sorted_dist_eng))])
y_eng = sorted_dist_eng[:,1].astype(np.float)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_eng,y_eng)
line = slope * x_eng + intercept

plt.plot(x_eng, y_eng, 'x', x_eng, line)
plt.xlabel("rank")
plt.ylabel("frequency")
plt.savefig("zipf_eng.png")

print("Slope:", slope, "Intercept:", intercept, "R:", r_value, "P:", p_value, "Error:", std_err)

#%%
x_hin = np.asarray([math.log10(i+1) for i in range(len(sorted_dist_hin))])
y_hin = sorted_dist_hin[:,1].astype(np.float)
slope, intercept, r_value, p_value, std_err = stats.linregress(x_hin,y_hin)
line = slope * x_hin + intercept

plt.plot(x_hin, y_hin, 'x', x_hin, line)
plt.xlabel("rank")
plt.ylabel("frequency")
plt.savefig("zipf_hin.png")

print("Slope:", slope, "Intercept:", intercept, "R:", r_value, "P:", p_value, "Error:", std_err)

#%%
