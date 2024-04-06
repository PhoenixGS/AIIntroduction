import os
import json
from tqdm import tqdm

words = {}
with open("./data/拼音汉字表/一二级汉字表.txt", "r", encoding="gbk") as file:
    # print(list(file.read().strip()))
    for voc in list(file.read().strip()):
        words[voc] = []
    # words = dict.fromkeys(list(file.read().strip()), [].copy())

with open("./data/拼音汉字表/拼音汉字表.txt", "r", encoding="gbk") as file:
    for line in file:
        pron = line.strip().split(' ')[0]
        for voc in line.strip().split(' ')[1:]:
            if voc in words.keys():
                words[voc].append(pron)

path = './data/语料库/sina_news_gbk'
lis = os.listdir(path)
print(lis)

one_word = {}
two_word = {}
tri_word = {}

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
                                one_word[pron] = {}
                            if not vocs[i] in one_word[pron].keys():
                                one_word[pron][vocs[i]] = 0
                            one_word[pron][vocs[i]] += 1
                for i in range(len(vocs) - 1):
                    if vocs[i] in words.keys() and vocs[i + 1] in words.keys():
                        voc = vocs[i] + ' ' + vocs[i + 1]
                        for pron1 in words[vocs[i]]:
                            for pron2 in words[vocs[i + 1]]:
                                pron = pron1 + ' ' + pron2
                                if not pron in two_word:
                                    two_word[pron] = {}
                                if not voc in two_word[pron].keys():
                                    two_word[pron][voc] = 0
                                two_word[pron][voc] += 1
                for i in range(len(vocs) - 2):
                    if vocs[i] in words.keys() and vocs[i + 1] in words.keys() and vocs[i + 2] in words.keys():
                        voc = vocs[i] + ' ' + vocs[i + 1] + ' ' + vocs[i + 2]
                        for pron1 in words[vocs[i]]:
                            for pron2 in words[vocs[i + 1]]:
                                for pron3 in words[vocs[i + 2]]:
                                    pron = pron1 + ' ' + pron2 + ' ' + pron3
                                    if not pron in tri_word.keys():
                                        tri_word[pron] = {}
                                    if not voc in tri_word[pron].keys():
                                        tri_word[pron][voc] = 0
                                    tri_word[pron][voc] += 1

with open("1_word.txt", "w", encoding="utf-8") as file:
    json.dump(one_word, file, ensure_ascii=False, indent=4)

with open("2_word.txt", "w", encoding="utf-8") as file:
    json.dump(two_word, file, ensure_ascii=False, indent=4)

with open("3_word.txt", "w", encoding="utf-8") as file:
    json.dump(tri_word, file, ensure_ascii=False, indent=4)