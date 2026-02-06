import numpy as np
import os
import pandas as pd

data_dir = os.path.join('data', 'ukb_simulated_data')
labels_path = os.path.join(data_dir, 'labels.csv')
train_path = os.path.join(data_dir, 'train.bin')

# 1. Load Labels
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 2. Identify Target Token IDs
def find_ids(keyword):
    return {i: labels[i] for i, l in enumerate(labels) if keyword.lower() in l.lower()}

bmi_ids = find_ids('BMI')
smoke_ids = find_ids('Smok')
alco_ids = find_ids('Alco')

# 3. Scan train.bin
data = np.fromfile(train_path, dtype=np.uint16)
num_patients = len(data) // 48
# Reshape to [Patient, Time]
records = data[:num_patients*48].reshape((num_patients, 48))

def get_stats(target_dict, name):
    print(f"\n--- {name} Distribution ---")
    all_hits = []
    for tid in target_dict.keys():
        # Count patients who have this token anywhere in their history
        count = (records == tid).any(axis=1).sum()
        all_hits.append({"Label": target_dict[tid], "Patients": count, "Prevalence": f"{(count/num_patients)*100:.1f}%"})
    print(pd.DataFrame(all_hits))

get_stats(bmi_ids, "BMI")
get_stats(smoke_ids, "Smoking")
get_stats(alco_ids, "Alcohol")