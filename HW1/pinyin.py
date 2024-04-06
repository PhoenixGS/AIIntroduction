import json
from math import log

# set parameter
lamb = 0.999999999
# alpha = 1000

# read data
# pron = {}
# with open("./word2pinyin.txt", 'r') as file:
#     for lines in file:
#         pron[lines.strip().split(' ')[0]] = lines.strip().split(' ')[1]

one_words = {}
two_words = {}
#cf = {}

se = set()

with open("./1_word.txt", "r") as file:
    one_word = json.load(file)

# maxx = 0
for (key, value) in one_word.items():
    #s = sum(value["counts"])
    for i in range(len(value["counts"])):
#        if not value["words"][i] in cf.keys():
#            cf[value["words"][i]] = {}
#        cf[value["words"][i]][key] = value["counts"][i]
#        if value["words"][i] == 
    #    value["counts"][i] /= s
        se.add(value["words"][i])
    # maxx = max(maxx, len(value["words"]))
    one_words[key] = dict(zip(value["words"], value["counts"]))

#print(cf['大'])
#for (key, value) in cf.items():
#    s = sum(value.values())
#    for k in value.keys():
#        value[k] /= s

#print(cf['不'])

with open("./2_word.txt", "r") as file:
    two_word = json.load(file)

# sss = 0
for (key, value) in two_word.items():
    #s = sum(value["counts"])
    #for i in range(len(value["counts"])):
    #    value["counts"][i] /= s
    two_words[key] = dict(zip(value["words"], value["counts"]))
    # sss += len(value["words"])

# print(one_words["ni"])

# print(two_word)
# ssss = 0
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
            now.append(-log(value))
            # now.append(0)
        f.append(now.copy())
        g.append([])
        # ssss += len(lis)

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
                        # print(word1, word2, p1)
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
            # for i in range(len(now)):
                # if now[i] > alpha:
                #     now[i] = 1000000000
                # now[i] = now[i] + log(len(w1))
            f.append(now.copy())
            g.append(pre.copy())

        
        # print(f)

        # print(g)
        
        n = len(lis) - 1
        ind = 0
        for i in range(len(f[n])):
            if f[n][i] < f[n][ind]:
                ind = i

        ans = [one_word[lis[n]]["words"][ind]]
        while n > 0:
            ind = g[n][ind]
            n = n - 1
            ans.append(one_word[lis[n]]["words"][ind])



        # ans = [one_word[lis[n]]["words"][ind]]
        # while n > 0:
        #     ind = g[n][ind]
        #     n = n - 1
        #     ans.append(one_word[lis[n]]["words"][ind])
        
        print(''.join(ans[::-1]))

except EOFError:
    pass

# print(ssss, maxx)
# print(sss)

# print(count)