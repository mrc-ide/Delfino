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

# MODE Options: 
# 'manual' (Default): Uses the competing risks race loop directly on x and a.
# 'automatic': Uses the model.generate() wrapper on cloned x and a.
MODE = 'manual'

DATA_DIR = os.path.join('data', 'ukb_simulated_data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.bin')
LABELS_PATH = os.path.join(DATA_DIR, 'labels.csv')
CKPT_PATH = 'out-delfino-baseline/ckpt.pt'

T_DEATH_ID = 1269

def generate_trajectories():

    # 1. LOAD INPUTS
    # load Token labels
    with open(LABELS_PATH, 'r') as f:
        labels_list = [line.strip() for line in f.readlines()]

    # Define Target Diseases and map to Model IDs (Raw Index + 1)
    # target_codes = ['E10', 'I50', 'I21', 'I63', 'N18']
    # target_map = {code: i + 1 for code in target_codes 
    #               for i, lbl in enumerate(labels_list) if lbl.startswith(code)}
    # all_metrics = []

    # load model checkpoint (weights)
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model = Delphi(DelphiConfig(**checkpoint['model_args'])).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Load training data as uint32 triplets [PID, Age, Token] in 3-columns
    train_data = np.fromfile(TRAIN_PATH, dtype=np.uint32).reshape(-1, 3)
    # Get the patient to index mapping.
    p2i = get_p2i(train_data)

    # create container for results
    trajectories = {}
    print(f"Running generation in {MODE} mode...")

    # Begin person loop
    for pid in tqdm(range(START_ID, END_ID)):
        
        # skip if specific person id is out of range of data.
        if pid >= len(p2i): continue
            
        # Get person's input context using official batching logic (includes +1 shift)
        x, a, _, _ = get_batch(ix=[pid], data=train_data, p2i=p2i, select='left', 
                              block_size=48, device=DEVICE, padding='random', no_event_token_rate=5)
        
        # Begin caculating trajectories
        if MODE == 'manual':
            # Direct pass: no cloning, uses the original tensors from get_batch
            curr_x, curr_a = x, a 
            manual_tokens = []
            manual_ages = []

            for _ in range(MAX_NEW_TOKENS):
                with torch.no_grad():
                    # Forward pass to get Hazard Rates (logits)
                    out = model(curr_x, age=curr_a)
                    logits = out[0][:, -1, :] 

                    # Competing Risks Race: Sample wait times from exponential distribution
                    # Inverse CDF method: T = -1/lambda * ln(U)
                    t_wait = torch.clamp(-torch.exp(-logits) * torch.rand(logits.shape, device=DEVICE).log(), min=0)
                    t_next = t_wait.min(1) # [0] is time, [1] is index
                    
                    next_id = t_next[1][:, None]
                    next_age = curr_a[..., [-1]] + t_next[0][:, None]

                manual_tokens.append(next_id.item())
                manual_ages.append(next_age.item())

                # Update context for the next step in the loop
                curr_x = torch.cat([curr_x, next_id], dim=1)
                curr_a = torch.cat([curr_a, next_age], dim=1)

                # Stop if the winner is Death
                if next_id.item() == T_DEATH_ID:
                    break
            
            # For manual, the full trajectory is now in the updated curr_x/curr_a
            gen_tokens = curr_x[0].cpu().numpy()
            gen_ages = curr_a[0].cpu().numpy()
            input_len = x.shape[1]

        elif MODE == 'automatic':
            # Wrapper pass: Clones the tensors so the original x and a are preserved
            with torch.no_grad():
                y, b, _ = model.generate(x.clone(), a.clone(), 
                                         max_new_tokens=MAX_NEW_TOKENS, 
                                         termination_tokens=[T_DEATH_ID])
            gen_tokens = y[0].cpu().numpy()
            gen_ages = b[0].cpu().numpy()
            input_len = x.shape[1]

        # 4. UNIFIED DISPLAY LOGIC (No-Shift interpretation)
        lines = ["Input trajectory:"]
        for i in range(len(gen_tokens)):
            # Divider between History and Generated Future
            if i == input_len:
                lines.append("=====================")
                lines.append(f"{MODE.capitalize()} Generated trajectory:")
            
            tid = int(gen_tokens[i])
            age_y = gen_ages[i] / DAYS_PER_YEAR

            # SKIP PADDING: Fixes the "weirdness" at the start of trajectories
            if tid == 0: 
                continue 
            
            # Map ID 1:1 to labels.csv index
            event_name = labels_list[tid] if tid < len(labels_list) else f"Unknown({tid})"
            lines.append(f"{age_y:2.1f}: {event_name}")
            
            # Stop display if terminal token is reached
            if i >= input_len and tid == T_DEATH_ID:
                break
                
        trajectories[str(pid)] = "\n".join(lines)

    # 5. SAVE FINAL CSV
    output_filename = f"temp_{MODE}_{START_ID}_{END_ID}_trajectories.csv"
    pd.DataFrame([trajectories]).to_csv(output_filename, index=False)
    print(f"\nDone. {MODE.capitalize()} results saved to {output_filename}")

if __name__ == "__main__":
    generate_trajectories()