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
END_ID = 300 # max of 7143
MAX_NEW_TOKENS = 100
DAYS_PER_YEAR = 365.25
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- INTERVENTION SETTINGS ---
# APPLY_INTERVENTION = True  
APPLY_INTERVENTION = False  
# Map ICD-10 Code -> Hazard Ratio (HR)
# e.g., {"I10": 0.5} means 50% reduction in Hypertension incidence
affected_diseases = {
    "I10": 0.001,
    # "I50": 0.8, # You can add multiple effects here
}

# MODE Options: 
# 'manual' (Default): Uses the competing risks race loop directly on x and a.
# 'automatic': Uses the model.generate() wrapper on cloned x and a.
MODE = 'manual'
# MODE = 'automatic' # note this currently has no tracking of disease timings, as can't easily input efficacies/logit biases

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

    # Identify all ICD-10 codes (Letter followed by numbers)
    # Mapping: {TokenID: "Code"}
    TRACKED_CODES = {}
    for i, label in enumerate(labels_list):
        # Match codes like I50, E11, but skip 'Padding' or 'No event'
        if len(label) >= 3 and label[0].isalpha() and label[1].isdigit():
            # Extract just the code part (e.g., "I50" from "I50 (heart failure)")
            code = label.split(' ')[0]
            TRACKED_CODES[i] = code

    # Distinct list of unique codes for CSV columns
    unique_codes = sorted(list(set(TRACKED_CODES.values())))

    # Vocabulary size is len(labels_list)
    logit_bias_vector = torch.zeros(len(labels_list), device=DEVICE)

    if APPLY_INTERVENTION:
        print(f"Applying Intervention on {len(affected_diseases)} disease(s):")
        # Reverse map to find indices for the affected codes
        code_to_id = {v: k for k, v in TRACKED_CODES.items()}
        
        for code, hr in affected_diseases.items():
            if code in code_to_id:
                tid = code_to_id[code]
                bias = np.log(hr)
                logit_bias_vector[tid] = bias
                print(f" - {code} (ID: {tid}): HR={hr} (Logit Bias: {bias:.4f})")
            else:
                print(f" - Warning: {code} not found in labels.")

    # Container for quantitative results (as opposed to string trajectories)
    all_metrics = []

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
                              block_size=128, device=DEVICE, padding='random', no_event_token_rate=5)
        
        # Initialize record with -1.0 (Absence)
        # SimulationStartAge is the age of the very last token in history
        start_age_y = a[0, -1].item() / DAYS_PER_YEAR
        inc_record = {
            "PatientID": pid, 
            "SimulationStartAge": start_age_y,
            **{code: -1.0 for code in unique_codes}
        }

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

                    # --- VECTORIZED INTERVENTION GATE ---
                    if APPLY_INTERVENTION:
                        # Adding the vector (mostly zeros) to the logits
                        logits += logit_bias_vector
                    # ------------------------------------

                    # Competing Risks Race: Sample wait times from exponential distribution
                    # Inverse CDF method: T = -1/lambda * ln(U)
                    t_wait = torch.clamp(-torch.exp(-logits) * torch.rand(logits.shape, device=DEVICE).log(), min=0)
                    t_next = t_wait.min(1) # [0] is time, [1] is index
                    
                    next_id = t_next[1][:, None]
                    next_age = curr_a[..., [-1]] + t_next[0][:, None]

                    # TRACKING: If token is a disease and not yet recorded, save the age
                    token_id = next_id.item()
                    if token_id in TRACKED_CODES:
                         code = TRACKED_CODES[token_id]
                         if inc_record[code] == -1.0:
                             inc_record[code] = next_age.item() / DAYS_PER_YEAR

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
                
        # add this person's lines to trajectories
        trajectories[str(pid)] = "\n".join(lines)

        # Append the record to the master list after each patient is done
        all_metrics.append(inc_record)

    # 5. Save Outputs

    status = "treated" if APPLY_INTERVENTION else "control"

    # Save string trajectories
    trajectories_output_filename = f"temp_{MODE}_{status}_{START_ID}_{END_ID}_trajectories.csv"
    df_results = pd.DataFrame(list(trajectories.items()), columns=["PatientID", "Trajectory"])
    df_results.to_csv(trajectories_output_filename, index=False)

    # Save the Incidence CSV
    incidence_filename = f"temp_{MODE}_{status}_{START_ID}_{END_ID}_incidence.csv"
    df_incidence = pd.DataFrame(all_metrics)
    # Reorder columns to put PatientID and StartAge first
    cols = ["PatientID", "SimulationStartAge"] + unique_codes
    df_incidence = df_incidence[cols]
    df_incidence.to_csv(incidence_filename, index=False)

    print(f"\nDone. Results saved with status '{status}' to:")
    print(f" - {trajectories_output_filename}")
    print(f" - {incidence_filename}")

if __name__ == "__main__":
    generate_trajectories()