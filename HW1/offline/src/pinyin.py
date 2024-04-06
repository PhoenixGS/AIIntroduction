import json
from math import log

# set parameter
lamb = 0.999999999
one_words = {}
two_words = {}

se = set()

with open("./1_word.txt", "r") as file:
    one_words = json.load(file)

for (key, value) in one_words.items():
    for word in value.keys():
        se.add(word)

with open("./2_word.txt", "r") as file:
    two_words = json.load(file)

count = 0
try:
    while True:
        lis = input().strip().split(' ')
        if lis == ['']:
            print()
            continue
        f = []
        g = []
        
        now = []
        for (key, value) in one_words[lis[0]].items():
            now.append(-log(value / len(se)))
        f.append(now.copy())
        g.append([])

        for i in range(1, len(lis)):
            now = []
            pre = []

            w1 = one_words[lis[i - 1]]
            w2 = one_words[lis[i]]
            if ' '.join(lis[i - 1: i + 1]) in two_words.keys():
                w12 = two_words[' '.join(lis[i - 1: i + 1])]
            else:
                w12 = None
            
            for (k, (word2, value2)) in enumerate(w2.items()):
                flag = False
                for (j, (word1, value1)) in enumerate(w1.items()):
                    count += 1
                    if w12 != None and ' '.join([word1, word2]) in w12.keys():
                        p1 = lamb * w12[' '.join([word1, word2])] / value1
                    else:
                        p1 = 0
                    p1 = p1 + (1 - lamb) * value2 / len(se)
                    ss = f[i - 1][j] - log(p1)
                    if not flag:
                        flag = True
                        now.append(ss)
                        pre.append(j)
                    else:
                        if ss < now[-1]:
                            now[-1] = ss
                            pre[-1] = j
            f.append(now.copy())
            g.append(pre.copy())
        
        n = len(lis) - 1
        ind = 0
        for i in range(len(f[n])):
            if f[n][i] < f[n][ind]:
                ind = i

        ans = [list(one_words[lis[n]].keys())[ind]]
        while n > 0:
            ind = g[n][ind]
            n = n - 1
            ans.append(list(one_words[lis[n]].keys())[ind])
        
        print(''.join(ans[::-1]))

except EOFError:
    pass