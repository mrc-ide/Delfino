import numpy as np
import pandas as pd
import os

# 1. Setup Paths
input_file = 'data/ukb_simulated_data/train.bin'
output_csv = 'data/ukb_simulated_data/preview_raw_data.csv'

# 2. Load and Reshape
BLOCK_SIZE = 48
data = np.fromfile(input_file, dtype=np.uint16)
num_patients = len(data) // BLOCK_SIZE
patients = data[:num_patients * BLOCK_SIZE].reshape(num_patients, BLOCK_SIZE)

# 3. Take the first 500 patients
subset = patients[:500, :]

# 4. Create Column Names (Time_0, Time_1, ..., Time_47)
column_names = [f'Step_{i}' for i in range(BLOCK_SIZE)]

# 5. Convert to a Pandas DataFrame and Save
df = pd.DataFrame(subset, columns=column_names)
df.index.name = 'Patient_ID'
df.to_csv(output_csv)

print(f"Successfully exported first 500 patients to {output_csv}")