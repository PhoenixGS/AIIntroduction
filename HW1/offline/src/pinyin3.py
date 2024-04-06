import json
from math import log
import sys

if "--full" in sys.argv:
    tri_path = "./3_word.txt"
else:
    tri_path = "./3_simple_word.txt"

# set parameter
lamb = 0.000000000999999
gamma = 0.999999999

one_words = {}
two_words = {}
tri_words = {}

se = set()

with open("./1_word.txt", "r") as file:
    one_words = json.load(file)

for (key, value) in one_words.items():
    for word in value.keys():
        se.add(word)

with open("./2_word.txt", "r") as file:
    two_words = json.load(file)

with open(tri_path, "r") as file:
    tri_words = json.load(file)

try:
    while True:
        lis = input().strip().split(' ')
        if lis == ['']:
            print()
            continue
        f = [[]]
        g = [[]]

        now = [[0 for k in range(len(one_words[lis[1]]))] for j in range(len(one_words[lis[0]]))]
        ww = two_words[' '.join([lis[0], lis[1]])]
        for (j, (key1, value1)) in enumerate(one_words[lis[0]].items()):
            for (k, (key2, value2)) in enumerate(one_words[lis[1]].items()):
                p = 0
                if ' '.join([key1, key2]) in ww.keys():
                    p += (lamb + gamma) * ww[' '.join([key1, key2])] / len(se) / len(se)
                p += (1 - lamb - gamma) * value1 * value2 / len(se) / len(se)
                now[j][k] = -log(p)
        f.append(now.copy())
        g.append([])

        for i in range(2, len(lis)):

            w1 = one_words[lis[i - 2]]
            w2 = one_words[lis[i - 1]]
            w3 = one_words[lis[i]]

            now = [[1e100 for l in range(len(w3))] for k in range(len(w2))]
            pre = [[1e100 for l in range(len(w3))] for k in range(len(w2))]

            if ' '.join(lis[i - 2: i]) in two_words.keys():
                w12 = two_words[' '.join(lis[i - 2: i])]
            else:
                w12 = None

            if ' '.join(lis[i - 1: i + 1]) in two_words.keys():
                w23 = two_words[' '.join(lis[i - 1: i + 1])]
            else:
                w23 = None
            
            if ' '.join(lis[i - 2: i + 1]) in tri_words.keys():
                w123 = tri_words[' '.join(lis[i - 2: i + 1])]
            else:
                w123 = None
            
            for (l, (word3, value3)) in enumerate(w3.items()):
                for (k, (word2, value2)) in enumerate(w2.items()):
                    for (j, (word1, value1)) in enumerate(w1.items()):
                        p1 = 0
                        if w123 != None and ' '.join([word1, word2, word3]) in w123.keys():
                            p1 += gamma * w123[' '.join([word1, word2, word3])] / w12[' '.join([word1, word2])]
                        if w23 != None and ' '.join([word2, word3]) in w23.keys():
                            p1 += lamb * w23[' '.join([word2, word3])] / value2
                        p1 = p1 + (1 - lamb - gamma) * value3 / len(se)
                        ss = f[i - 1][j][k] - log(p1)
                        if ss < now[k][l]:
                            now[k][l] = ss
                            pre[k][l] = j
            f.append(now.copy())
            g.append(pre.copy())
        
        n = len(lis) - 1
        indj = 0
        indk = 0
        for j in range(len(f[n])):
            for k in range(len(f[n][j])):
                if f[n][j][k] < f[n][indj][indk]:
                    indj = j
                    indk = k
                
        ans = [list(one_words[lis[n]].keys())[indk]]
        while n > 1:
            t = g[n][indj][indk]
            indk = indj
            indj = t
            n = n - 1
            ans.append(list(one_words[lis[n]].keys())[indk])
        ans.append(list(one_words[lis[0]].keys())[indj])
        
        print(''.join(ans[::-1]))

except EOFError:
    pass