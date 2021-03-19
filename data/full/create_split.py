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
    with open(f"{split}.csv", "w") as out_file:
        for cancer in LABELS.keys():
            with open(f"../raw_splits/{cancer}_{split}.csv", "r") as in_file:
                for row in in_file:
                    out_file.write(f"{row.strip()}\n")        
