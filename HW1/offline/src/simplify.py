import json
from tqdm import tqdm

with open("./3_word.txt", "r") as file:
    tri_words = json.load(file)

for (key, value) in tqdm(tri_words.items()):
    lis = []
    for (k, v) in value.items():
        if v <= 10:
            lis.append(k)
    for k in lis:
        value.pop(k)

with open("./3_simple_word.txt", "w") as file:
    json.dump(tri_words, file, ensure_ascii=False, indent=4)
