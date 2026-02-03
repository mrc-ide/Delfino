import pickle
import os

# Define path to metadata
meta_path = os.path.join('data', 'ukb_simulated_data', 'meta.pkl')

if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    # Print first 100 tokens to be sure we find what we need
    print("--- Vocabulary Sample (First 100 Tokens) ---")
    tokens = list(meta['stoi'].keys())
    for i, token in enumerate(tokens[:100]):
        print(f"{i}: {token}")
else:
    print(f"Error: Could not find {meta_path}. Check your folder structure!")