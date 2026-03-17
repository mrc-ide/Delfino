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

        # with torch.no_grad():
        #     # Single forward pass to get baseline probabilities for the next event
        #     out = model(x, age=a)
        #     logits_baseline = out[0][0, -1, :] # Logits for the token following history
        #     probs_baseline = torch.softmax(logits_baseline, dim=-1)

        #     # Extract probabilities for target Model IDs
        #     metrics = {"ID": pid}
        #     for code, m_tid in target_map.items():
        #         metrics[f"prob_{code}"] = probs_baseline[m_tid].item()
        #     all_metrics.append(metrics)

        # # --- MANUAL SAMPLING (THE COMPETING RISKS RACE) ---
        # # Initialize the generation context with history from get_batch
        
        #     manual_traj_tokens = []
        #     manual_traj_ages = []

        # for _ in range(MAX_NEW_TOKENS):

        #     with torch.no_grad():
        #         # 1. Calculate Hazard Rates (Logits)
        #         out = model(x, age=a)
        #         logits = out[0][:, -1, :] # Hazards for the next step

        #         # 2. The Exponential Race (Inverse CDF sampling)
        #         # t_wait is the time until each potential event happens
        #         t_wait = torch.clamp(-torch.exp(-logits) * torch.rand(logits.shape, device=DEVICE).log(), min=0)
                
        #         # Find the winner: shortest wait time
        #         # t_next[0] is the wait time, t_next[1] is the token ID
        #         t_next = t_wait.min(1)
                
        #         next_token_id = t_next[1][:, None]
        #         next_age = a[..., [-1]] + t_next[0][:, None]

        #     # Store the winner
        #     manual_traj_tokens.append(next_token_id.item())
        #     manual_traj_ages.append(next_age.item())

        #     # Update the context for the next step in the loop
        #     x = torch.cat([x, next_token_id], dim=1)
        #     a = torch.cat([a, next_age], dim=1)

        #     # TERMINATION: Break if the winner is Death (1269)
        #     if next_token_id.item() == 1269:
        #         break
        
        # # Format the manual trajectory string (following your existing logic)
        # #hist_raw = (x[0].cpu().numpy() - 1)
        # hist_raw = (x[0].cpu().numpy())
        # hist_ages = a[0].cpu().numpy() / DAYS_PER_YEAR
        
        # m_lines = ["Input trajectory:"]
        # for t, age in zip(hist_raw, hist_ages):
        #     #if t < 0: continue # Skip padding
        #     if t == 0: continue # Only skip padding
        #     m_lines.append(f"{age:2.1f}: {labels_list[t]}")
        
        # m_lines.append("=====================")
        # m_lines.append("Manual Generated trajectory:")
        
        # for t, age_days in zip(manual_traj_tokens, manual_traj_ages):
        #     age_y = age_days / DAYS_PER_YEAR
        #     m_lines.append(f"{age_y:2.1f}: {labels_list[t]}")
        #     if t == T_DEATH_ID: break
            
        # manual_results[str(pid)] = "\n".join(m_lines)

        # with torch.no_grad():
        #     # generate() returns [Input + Generated]
        #     y, b, _ = model.generate(x, a, max_new_tokens=MAX_NEW_TOKENS, termination_tokens=[T_DEATH_ID])

        # full_tokens_model = y[0].cpu().numpy()
        # full_ages_years = b[0].cpu().numpy() / DAYS_PER_YEAR
        
        # # History window size is 48
        # input_len = x.shape[1] 
        # lines = ["Input trajectory:"]

        # for i in range(len(full_tokens_model)):
        #     # BUG FIX: Check for divider BEFORE skipping padding
        #     if i == input_len:
        #         lines.append("=====================")
        #         lines.append("Generated trajectory:")
            
        #     model_tid = full_tokens_model[i]
        #     #raw_tid = model_tid - 1
        #     raw_tid = model_tid
        #     age = full_ages_years[i]

        #     # Filter logic: 
        #     # Skip model-level padding (0) or labels.csv padding (raw 0)
        #     if model_tid == 0 or raw_tid == 0:
        #         continue

        #     event_name = labels_list[raw_tid] if 0 <= raw_tid < len(labels_list) else f"Unknown({raw_tid})"
        #     lines.append(f"{age:2.1f}: {event_name}")
            
        #     # Stop sequence if model predicts death
        #     if i >= input_len and raw_tid == T_DEATH_RAW_ID:
        #         break
                
        # trajectories[str(pid)] = "\n".join(lines)

    # # Save as transposed CSV
    # pd.DataFrame([trajectories]).to_csv(f"temp_base_{START_ID}_{END_ID}_trajectories.csv", index=False)
    # print(f"\nSaved trajectories to temp_base_{START_ID}_{END_ID}_trajectories.csv")

    # # Save the quantitative probabilities
    # df_probs = pd.DataFrame(all_metrics)
    # df_probs.to_csv(f"temp_base_{START_ID}_{END_ID}_results.csv", index=False)
    # print(f"Probabilities saved to temp_base_{START_ID}_{END_ID}_results.csv")

    # # Save the manual race trajectories
    # df_manual = pd.DataFrame([manual_results])
    # manual_output_name = f"temp_manual_{START_ID}_{END_ID}_trajectories.csv"
    # df_manual.to_csv(manual_output_name, index=False)
    # print(f"Manual trajectories saved to: {manual_output_name}")

if __name__ == "__main__":
    generate_trajectories()