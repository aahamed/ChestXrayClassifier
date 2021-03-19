import csv
import random

LABELS = {
    "Atelectasis": 0,
    "Consolidation": 1,
    "Infiltration": 2,
    "Pneumothorax": 3,
    "Edema": 4,
    "Emphysema": 5,
    "Fibrosis": 6,
    "Effusion": 7,
    "Pneumonia": 8,
    "Pleural_Thickening": 9,
    "Cardiomegaly": 10,
    "Nodule": 11,
    "Mass": 12,
    "Hernia": 13,
    "No Finding": 14,
}

splits = ["train", "val", "test"]
for split in splits:
    cancer_splits = {cancer: list() for cancer in LABELS.keys()}

    for cancer in LABELS.keys():
        with open(f"../raw_splits/{cancer}_{split}.csv", "r") as cancer_file:
            for row in cancer_file:
                cancer_splits[cancer].append(row.strip())
        random.shuffle(cancer_splits[cancer])

    cancer_counts = [(len(cancer_splits[cancer]), cancer) for cancer in cancer_splits.keys()]
    cancer_counts.sort()
    min_count = cancer_counts[0][0]
    print(f"Minimum count {min_count} for '{cancer_counts[0][1]}'")

    cur_split = list()
    for cancer in cancer_splits.keys():
        cur_split += cancer_splits[cancer][:min_count]

    with open(f"./{split}.csv", "w") as out_file:
        out_file.write("\n".join(cur_split))
        
