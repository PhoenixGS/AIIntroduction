import json
from math import log

# set parameter
lamb = 0.000000000999999
gamma = 0.999999999
# alpha = 1000

# read data
# pron = {}
# with open("./word2pinyin.txt", 'r') as file:
#     for lines in file:
#         pron[lines.strip().split(' ')[0]] = lines.strip().split(' ')[1]

one_words = {}
two_words = {}
tri_words = {}
#cf = {}

se = set()

with open("./1_word.txt", "r") as file:
    one_words = json.load(file)

# maxx = 0
for (key, value) in one_words.items():
#     #s = sum(value["counts"])
    for word in value.keys():
# #        if not value["words"][i] in cf.keys():
# #            cf[value["words"][i]] = {}
# #        cf[value["words"][i]][key] = value["counts"][i]
# #        if value["words"][i] == 
#     #    value["counts"][i] /= s
        se.add(word)
#     # maxx = max(maxx, len(value["words"]))

#print(cf['大'])
#for (key, value) in cf.items():
#    s = sum(value.values())
#    for k in value.keys():
#        value[k] /= s

#print(cf['不'])

with open("./2_word.txt", "r") as file:
    two_words = json.load(file)

with open("./3_word.txt", "r") as file:
    tri_words = json.load(file)

# for (key, value) in one_words.items():
#     s = sum(value.values())
#     for k in value.keys():
#         value[k] /= s

# for (key, value) in two_words.items():
#     s = sum(value.values())
#     for k in value.keys():
#         value[k] /= s

# for (key, value) in tri_words.items():
#     s = sum(value.values())
#     for k in value.keys():
#         value[k] /= s



# print("read over")

# sss = 0
# for (key, value) in two_words.items():
    #s = sum(value["counts"])
    #for i in range(len(value["counts"])):
    #    value["counts"][i] /= s
    # two_words[key] = dict(zip(value["words"], value["counts"]))
    # sss += len(value["words"])

# print(one_words["ni"])

# print(two_word)
# ssss = 0
# count = 0
try:
    while True:
        lis = input().strip().split(' ')
        if lis == ['']:
            print()
            continue
        f = [[]]
        g = [[]]
        
        # now = []
        # for (key, value) in one_words[lis[0]].items():
        #     now.append(-log(value))
        #     # now.append(0)
        # f.append(now.copy())
        # g.append([])
        # ssss += len(lis)

        now = [[0 for k in range(len(one_words[lis[1]]))] for j in range(len(one_words[lis[0]]))]
        ww = two_words[' '.join([lis[0], lis[1]])]
        # ssww = sum(ww.values())
        for (j, (key1, value1)) in enumerate(one_words[lis[0]].items()):
            for (k, (key2, value2)) in enumerate(one_words[lis[1]].items()):
                p = 0
                if ' '.join([key1, key2]) in ww.keys():
                    p += (lamb + gamma) * ww[' '.join([key1, key2])] / len(se) / len(se)
                p += (1 - lamb - gamma) * value1 * value2 / len(se) / len(se)
                now[j][k] = -log(p)
                # print(key1, key2, p)
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

            
            # for (k, (word2, value2)) in enumerate(w2.items()):
            #     flag = False
            #     for (j, (word1, value1)) in enumerate(w1.items()):
            #         # count += 1
            #         if w12 != None and ' '.join([word1, word2]) in w12.keys():
            #             p1 = lamb * w12[' '.join([word1, word2])] / value1
            #             # print(word1, word2, p1)
            #         else:
            #             p1 = 0
            #         p1 = p1 + (1 - lamb) * value2 / len(se)
            #         ss = f[i - 1][j] - log(p1)
            #         if not flag:
            #             flag = True
            #             now.append(ss)
            #             pre.append(j)
            #         else:
            #             if ss < now[-1]:
            #                 now[-1] = ss
            #                 pre[-1] = j
            # # for i in range(len(now)):
            #     # if now[i] > alpha:
            #     #     now[i] = 1000000000
            #     # now[i] = now[i] + log(len(w1))
            # f.append(now.copy())
            # g.append(pre.copy())

        
        # print(f)

        # print(g)
        
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