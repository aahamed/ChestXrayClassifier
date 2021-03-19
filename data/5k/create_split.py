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

split_sizes = {
    "train": 5000,
    "val": 625,
    "test": 625
}

for split, size in split_sizes.items():
    cancer_splits = {cancer: list() for cancer in LABELS.keys()}

    for cancer in LABELS.keys():
        with open(f"../raw_splits/{cancer}_{split}.csv", "r") as cancer_file:
            for row in cancer_file:
                cancer_splits[cancer].append(row.strip())
        random.shuffle(cancer_splits[cancer])

    cancer_counts = [(len(cancer_splits[cancer]), cancer) for cancer in cancer_splits.keys()]
    cancer_counts.sort()
    
    print(f"Breakdown for {split}")
    remaining = len(cancer_counts)
    cur_split = list()
    for _, cancer in cancer_counts:
        ideal_size = int(size / remaining)
        actual_split = cancer_splits[cancer][:ideal_size]
        cur_split += actual_split
        size -= len(actual_split)
        remaining -= 1
        print(f"\t{cancer}: {len(actual_split)}")


    with open(f"./{split}.csv", "w") as out_file:
        out_file.write("\n".join(cur_split))
        
