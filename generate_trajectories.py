import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Official Delphi imports
from model import Delphi, DelphiConfig
from utils import get_p2i, get_batch

# --- SETTINGS ---
START_ID = 0
END_ID = 50
MAX_NEW_TOKENS = 100
DAYS_PER_YEAR = 365.25
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Path Configuration
DATA_DIR = os.path.join('data', 'ukb_simulated_data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.bin')
LABELS_PATH = os.path.join(DATA_DIR, 'labels.csv')
CKPT_PATH = 'out-delfino-baseline/ckpt.pt'

# The model's internal ID for Death (1269 + 1)
T_DEATH_MODEL_ID = 1270 

def generate_trajectories():
    # 1. Load Labels
    with open(LABELS_PATH, 'r') as f:
        labels_list = [line.strip() for line in f.readlines()]

    # 2. Load Model
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model = Delphi(DelphiConfig(**checkpoint['model_args'])).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 3. Load Data & Mapping
    train_data = np.fromfile(TRAIN_PATH, dtype=np.uint32).reshape(-1, 3)
    p2i = get_p2i(train_data)

    results = {}
    print(f"Generating trajectories using official get_batch logic...")

    for pid in tqdm(range(START_ID, END_ID)):
        # We use get_batch for a 'batch' of 1 patient
        # This handles: +1 shift, random 'No event' insertion, and lifestyle jittering
        x, a, _, _ = get_batch(
            ix=[pid], 
            data=train_data, 
            p2i=p2i, 
            select='left', 
            block_size=48, 
            device=DEVICE,
            padding='random',
            no_event_token_rate=5 # Standard Delphi rate: 1 event per 5 years
        )

        # Delphi model generate() call
        with torch.no_grad():
            y, b, _ = model.generate(
                x, 
                a, 
                max_new_tokens=MAX_NEW_TOKENS, 
                termination_tokens=[T_DEATH_MODEL_ID]
            )

        # --- DATA RECOVERY (Ground Truth Reverse Shift) ---
        # 1. Take the model's output tokens and subtract 1 to match labels.csv
        # 2. Filter out 0 (which was the model's padding/mask after the -1 shift)
        full_tokens_model = y[0].cpu().numpy().flatten()
        full_tokens_raw = full_tokens_model - 1
        full_ages_years = b[0].cpu().numpy().flatten() / DAYS_PER_YEAR
        
        # Length of history provided by get_batch (to find where generated part starts)
        # Note: get_batch might have added 'No event' tokens, increasing the length
        input_len = x.shape[1]

        lines = ["Input trajectory:"]
        for i, (tid, age) in enumerate(zip(full_tokens_raw, full_ages_years)):
            # Skip model padding (now -1)
            if tid < 0: continue
            
            if i == input_len:
                lines.append("=====================")
                lines.append("Generated trajectory:")
            
            event_name = labels_list[tid] if 0 <= tid < len(labels_list) else f"Unknown({tid})"
            lines.append(f"{age:2.1f}: {event_name}")
            
            if i >= input_len and tid == 1269: # Raw ID for Death
                break
                
        results[str(pid)] = "\n".join(lines)

    # 5. Save Final CSV
    df_out = pd.DataFrame([results])
    output_filename = f"temp_base_{START_ID}_{END_ID}_trajectories.csv"
    df_out.to_csv(output_filename, index=False)
    print(f"\nSuccess. Trajectories generated with full Delphi machinery.")

if __name__ == "__main__":
    generate_trajectories()