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

cancer_to_file = {label: list() for label in LABELS.keys()}
with open("../Data_Entry_2017_v2020.csv", "r") as csv_file:
    reader = csv.reader(csv_file)

    # Skip header
    reader.__next__()

    for row in reader:
        for cancer in row[1].split("|"):
            cancer_to_file[cancer].append(row[0])

for _, l in cancer_to_file.items():
    random.shuffle(l)

cancer_counts = [(len(cancer_to_file[cancer]), cancer) for cancer in cancer_to_file.keys()]
cancer_counts.sort()

used = list()
for count, cancer in cancer_counts:
    train_split = list()
    val_split = list()
    test_split = list()

    train_count = int(0.8 * count)
    val_count = int(0.1 * count)

    for f in cancer_to_file[cancer][0:train_count]:
        if f not in used:
            train_split.append(f)
            used.append(f)

    for f in cancer_to_file[cancer][train_count:train_count+val_count]:
        if f not in used:
            val_split.append(f)
            used.append(f)

    for f in cancer_to_file[cancer][train_count+val_count:]:
        if f not in used:
            test_split.append(f)
            used.append(f)

    with open(f"{cancer}_train.csv", "w") as train_file:
        train_file.write("\n".join(train_split))

    with open(f"{cancer}_val.csv", "w") as val_file:
        val_file.write("\n".join(val_split))

    with open(f"{cancer}_test.csv", "w") as test_file:
        test_file.write("\n".join(test_split))

print(cancer_counts)