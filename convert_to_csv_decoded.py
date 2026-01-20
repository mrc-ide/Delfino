import numpy as np
import pandas as pd
import os

# 1. Setup Paths
dataset_dir = 'data/ukb_simulated_data'
input_file = os.path.join(dataset_dir, 'train.bin')
labels_file = os.path.join(dataset_dir, 'Labels.csv')
output_csv = os.path.join(dataset_dir, 'human_readable_patients_v5.csv')

# 2. Load Labels (Comma-Safe)
with open(labels_file, 'r', encoding='utf-8') as f:
    label_list = [line.strip().replace('"', '') for line in f.readlines()]

# 3. Load & Reshape Binary Data (uint16)
BLOCK_SIZE = 48
data = np.fromfile(input_file, dtype=np.uint16)
num_patients = 500  # Subset for readability
patients = data[:num_patients * BLOCK_SIZE].reshape(num_patients, BLOCK_SIZE)

# 4. Decode
decoded_data = []
for row in patients:
    decoded_row = [label_list[int(t)] if int(t) < len(label_list) else f"ID_{t}" for t in row]
    decoded_data.append(decoded_row)

# 5. Save (The index=False prevents Excel from adding an extra column)
df = pd.DataFrame(decoded_data, columns=[f'Step_{i}' for i in range(BLOCK_SIZE)])
df.to_csv(output_csv, index_label='Patient_ID')
print(f"✅ Created: {output_csv}")