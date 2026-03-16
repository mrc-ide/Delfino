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

DATA_DIR = os.path.join('data', 'ukb_simulated_data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.bin')
LABELS_PATH = os.path.join(DATA_DIR, 'labels.csv')
CKPT_PATH = 'out-delfino-baseline/ckpt.pt'

T_DEATH_RAW_ID = 1269
T_DEATH_MODEL_ID = 1270 

def generate_trajectories():
    with open(LABELS_PATH, 'r') as f:
        labels_list = [line.strip() for line in f.readlines()]

    # Define Target Diseases and map to Model IDs (Raw Index + 1)
    target_codes = ['E10', 'I50', 'I21', 'I63', 'N18']
    target_map = {code: i + 1 for code in target_codes 
                  for i, lbl in enumerate(labels_list) if lbl.startswith(code)}
    all_metrics = []

    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model = Delphi(DelphiConfig(**checkpoint['model_args'])).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Load 3-column data
    train_data = np.fromfile(TRAIN_PATH, dtype=np.uint32).reshape(-1, 3)
    p2i = get_p2i(train_data)

    results = {}
    for pid in tqdm(range(START_ID, END_ID)):
        if pid >= len(p2i): continue
            
        # Get input context using official batching logic (includes +1 shift)
        x, a, _, _ = get_batch(ix=[pid], data=train_data, p2i=p2i, select='left', 
                              block_size=48, device=DEVICE, padding='random', no_event_token_rate=5)
        
        with torch.no_grad():
            # Single forward pass to get baseline probabilities for the next event
            out = model(x, age=a)
            logits_baseline = out[0][0, -1, :] # Logits for the token following history
            probs_baseline = torch.softmax(logits_baseline, dim=-1)

            # Extract probabilities for target Model IDs
            metrics = {"ID": pid}
            for code, m_tid in target_map.items():
                metrics[f"prob_{code}"] = probs_baseline[m_tid].item()
            all_metrics.append(metrics)

        with torch.no_grad():
            # generate() returns [Input + Generated]
            #y, b, _ = model.generate(x, a, max_new_tokens=MAX_NEW_TOKENS, termination_tokens=[T_DEATH_MODEL_ID])
            y, b, _ = model.generate(x, a, max_new_tokens=MAX_NEW_TOKENS, termination_tokens=[T_DEATH_RAW_ID])

        full_tokens_model = y[0].cpu().numpy()
        full_ages_years = b[0].cpu().numpy() / DAYS_PER_YEAR
        
        # History window size is 48
        input_len = x.shape[1] 
        lines = ["Input trajectory:"]

        for i in range(len(full_tokens_model)):
            # BUG FIX: Check for divider BEFORE skipping padding
            if i == input_len:
                lines.append("=====================")
                lines.append("Generated trajectory:")
            
            model_tid = full_tokens_model[i]
            #raw_tid = model_tid - 1
            raw_tid = model_tid
            age = full_ages_years[i]

            # Filter logic: 
            # Skip model-level padding (0) or labels.csv padding (raw 0)
            if model_tid == 0 or raw_tid == 0:
                continue

            event_name = labels_list[raw_tid] if 0 <= raw_tid < len(labels_list) else f"Unknown({raw_tid})"
            lines.append(f"{age:2.1f}: {event_name}")
            
            # Stop sequence if model predicts death
            if i >= input_len and raw_tid == T_DEATH_RAW_ID:
                break
                
        results[str(pid)] = "\n".join(lines)

    # Save as transposed CSV
    pd.DataFrame([results]).to_csv(f"temp_base_{START_ID}_{END_ID}_trajectories.csv", index=False)
    print(f"\nSaved trajectories to temp_base_{START_ID}_{END_ID}_trajectories.csv")

    # Save the quantitative probabilities
    df_probs = pd.DataFrame(all_metrics)
    df_probs.to_csv(f"temp_base_{START_ID}_{END_ID}_results.csv", index=False)
    print(f"Probabilities saved to temp_base_{START_ID}_{END_ID}_results.csv")

    

if __name__ == "__main__":
    generate_trajectories()