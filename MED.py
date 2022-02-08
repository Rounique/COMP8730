from numba import jit, cuda
import numpy as np
import time
import pickle
from nltk.corpus import wordnet


wordnet.all_synsets()

def dataset_preparation(datfile="missp.dat"):
    dataset = [i.strip() for i in open(datfile).readlines()]
    originalWord = []
    misspelledWord = []
    for token in dataset:
        if token[0]=='$':
            ORGtoken = token[1:]
        else:
            originalWord.append(ORGtoken.lower())
            misspelledWord.append(token.lower())
    return np.asarray(originalWord), np.asarray(misspelledWord)
def evaluation(predicted, ground_truth, K):
    hits = 0
    for count, gt in enumerate(ground_truth):
        if gt in predicted[count][:K]:
            hits += 1
    return hits/len(predicted)

@jit
def editDistance(str1, str2, m, n):
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i][j-1]+1,  
                                   dp[i-1][j]+1,       
                                   dp[i-1][j-1]+2) 
 
    return dp[m][n]

GroundTruth, Misspelled = dataset_preparation(datfile="missp.dat")
Dictionary = []
for i in wordnet.all_synsets():
    Dictionary.append(i.name().split('.')[0])
Dictionary = np.asarray(Dictionary)
Dictionary = np.unique(Dictionary)
K=10
TOPS = []

for mcidx,ms in enumerate(Misspelled):
    distances = []
    for cidx, c in enumerate(Dictionary):
        n = len(ms)
        m = len(c)
        distances.append(editDistance(ms,c,len(ms),len(c)))
    distances = np.asarray(distances)
    idx = distances.argpartition(range(K))[:K]
    kTops = Dictionary[np.array(idx)]
    TOPS.append(kTops)

top1 = evaluation(TOPS, GroundTruth, K=1)
top5 = evaluation(TOPS, GroundTruth, K=5)
top10 = evaluation(TOPS, GroundTruth, K=10)
print(top1, top5, top10)
