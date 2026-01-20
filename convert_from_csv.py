import numpy as np
import pandas as pd
import os

# 1. Setup Paths
dataset_dir = 'data/ukb_simulated_data'
input_csv = os.path.join(dataset_dir, 'human_readable_patients_v5.csv')
labels_file = os.path.join(dataset_dir, 'Labels.csv')
output_bin = os.path.join(dataset_dir, 'reconstructed_train.bin')

# 2. Load Reverse-Map (String -> Integer)
with open(labels_file, 'r', encoding='utf-8') as f:
    label_list = [line.strip().replace('"', '') for line in f.readlines()]
name_to_token = {name: idx for idx, name in enumerate(label_list)}

# 3. Load CSV (Using iloc to skip whatever the ID column is named)
df = pd.read_csv(input_csv)
medical_data = df.iloc[:, 1:] # Skips first column (Patient_ID)

# 4. Re-encode
re_encoded_rows = []
for _, row in medical_data.iterrows():
    token_row = []
    for val in row:
        val_str = str(val).strip()
        if val_str in name_to_token:
            token_row.append(name_to_token[val_str])
        elif "_" in val_str: # Handles "ID_9651" or "DIAG_ID_23451"
            token_row.append(int(val_str.split('_')[-1]))
        else:
            token_row.append(0)
    re_encoded_rows.append(token_row)

# 5. Save as raw uint16 Binary
final_array = np.array(re_encoded_rows, dtype=np.uint16)
final_array.tofile(output_bin)
print(f"✅ Success! Created: {output_bin} ({final_array.shape})")