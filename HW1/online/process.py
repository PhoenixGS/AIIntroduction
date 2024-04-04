import os
import json
from tqdm import tqdm

words = {}
with open("../拼音汉字表/一二级汉字表.txt", "r", encoding="gbk") as file:
    # print(list(file.read().strip()))
    for voc in list(file.read().strip()):
        words[voc] = []
    # words = dict.fromkeys(list(file.read().strip()), [].copy())

with open("../拼音汉字表/拼音汉字表.txt", "r", encoding="gbk") as file:
    for line in file:
        pron = line.strip().split(' ')[0]
        for voc in line.strip().split(' ')[1:]:
            if voc in words.keys():
                words[voc].append(pron)

path = '../语料库/sina_news_gbk'
lis = os.listdir(path)
print(lis)

one_word = {}
two_word = {}

for name in lis:
    if '-' in name and 'txt' in name:
        print("reading...", name)
        with open(os.path.join(path, name), "r", encoding="gbk") as file:
            for line in tqdm(file):
                vocs = list(line.strip())
                for i in range(len(vocs) - 1):
                    if vocs[i] in words.keys():
                        for pron in words[vocs[i]]:
                            if not pron in one_word:
                                one_word[pron] = {"words": [], "counts": []}
                            if not vocs[i] in one_word[pron]["words"]:
                                one_word[pron]["words"].append(vocs[i])
                                one_word[pron]["counts"].append(0)
                            one_word[pron]["counts"][one_word[pron]["words"].index(vocs[i])] += 1
                for i in range(len(vocs) - 1):
                    if vocs[i] in words.keys() and vocs[i + 1] in words.keys():
                        voc = vocs[i] + ' ' + vocs[i + 1]
                        for pron1 in words[vocs[i]]:
                            for pron2 in words[vocs[i + 1]]:
                                pron = pron1 + ' ' + pron2
                                if not pron in two_word:
                                    two_word[pron] = {"words": [], "counts": []}
                                if not voc in two_word[pron]["words"]:
                                    two_word[pron]["words"].append(voc)
                                    two_word[pron]["counts"].append(0)
                                two_word[pron]["counts"][two_word[pron]["words"].index(voc)] += 1
    # print(one_word)
    # print(two_word)
    with open("1_word.txt", "w", encoding="utf-8") as file:
        json.dump(one_word, file, ensure_ascii=False, indent=4)

    with open("2_word.txt", "w", encoding="utf-8") as file:
        json.dump(two_word, file, ensure_ascii=False, indent=4)