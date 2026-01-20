import pickle
with open('data/ukb_simulated_data/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
print("Keys found in meta.pkl:", meta.keys())

import numpy as np
data = np.fromfile('data/ukb_simulated_data/train.bin', dtype=np.uint16)
print(f"Max Token ID in data: {data.max()}")
print(f"Total Unique Tokens: {len(np.unique(data))}")