import numpy as np
import os

# 1. Setup Paths
# We use the raw data as input and save a new 'smoking' version
DATA_DIR = 'data/ukb_simulated_data'
INPUT_FILE = os.path.join(DATA_DIR, 'train.bin')
OUTPUT_FILE = os.path.join(DATA_DIR, 'train_smoking.bin')

# 2. Load the binary data
# uint16 is the standard 2-byte integer format for the Delphi/Delfino tokens
data = np.fromfile(INPUT_FILE, dtype=np.uint16)
print(f"Original file size: {len(data)} tokens.")

# 3. Fix the Reshape Error (The "Crop" Logic)
BLOCK_SIZE = 48
num_patients = len(data) // BLOCK_SIZE  # Floor division
clean_size = num_patients * BLOCK_SIZE  # The amount of data that fits perfectly

if len(data) % BLOCK_SIZE != 0:
    print(f"Note: Cropping {len(data) % BLOCK_SIZE} leftover tokens to fit 48-column grid.")

# 4. Reshape into (Number of Patients, 48 Timesteps)
patients = data[:clean_size].reshape(num_patients, BLOCK_SIZE)
print(f"Matrix shape created: {patients.shape} (Patients x Timesteps)")

# 5. The "Smoking" Surgery
# In the Delphi demo vocab: Token 4 = Current Smoker.
# We modify the first 1,000 patients. 
# We put the token at Index 2 (usually where lifestyle/baseline starts).
patients[:1000, 2] = 4 

# 6. Save the Modified Heir
patients.tofile(OUTPUT_FILE)
print(f"Success! Created: {OUTPUT_FILE}")