import os
import pickle
import torch
import numpy as np
import pandas as pd
from model import Delphi, DelphiConfig

# --- 1. CONFIGURATION & SEEDING ---
SEED_OFFSET = 42 
NUM_PATIENTS = 1000         # 22,661 max available in your synthetic data
TIME_HORIZON = 20         # Years to simulate
APPLY_INTERVENTION = True 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- NEW OUTPUT OPTIONS ---
PRINT_TRAJECTORIES = True    # Set to True to see the "Story" in console
SAVE_TRAJECTORY_FILES = True  # Set to True to write full histories to a .txt file

# --- 2. DYNAMIC LABEL LOOKUP (Robust Version) ---
labels_path = os.path.join('data', 'ukb_simulated_data', 'labels.csv')
with open(labels_path, 'r') as f:
    labels_list = [line.strip() for line in f.readlines()]

def get_id(label_name):
    try:
        return str(labels_list.index(label_name))
    except ValueError:
        return None

# Mapping "Delfino" concepts to actual labels found in your CSV
T_MALE = get_id('Male')
T_BMI_HIGH = get_id('BMI_high')
T_BMI_MID = get_id('BMI_mid')
T_DIABETES = get_id('E11 (non-insulin-dependent diabetes mellitus)')
T_HF = get_id('I50 (heart failure)')
T_STROKE = get_id('I63 (cerebral infarction)')
T_DEATH = get_id('Death')

# Economic values & Weights [FUDGE PLACEHOLDERS]
DISABILITY_WEIGHTS = {T_DIABETES: 0.05, T_HF: 0.18, T_STROKE: 0.23, T_DEATH: 1.0}
DISEASE_COSTS = {T_DIABETES: 2500, T_HF: 8000, T_STROKE: 15000}
GLP_YEARLY_COST = 1200

# --- 3. LOAD DELPHI WEIGHTS ---
out_dir = 'out-delfino-baseline'
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=DEVICE)

conf = DelphiConfig(**checkpoint['model_args'])
model = Delphi(conf)
model.load_state_dict(checkpoint['model'])
model.to(DEVICE).eval()

# Load Token Meta
meta_path = os.path.join('data', 'ukb_simulated_data', 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

# --- 4. SIMULATOR FUNCTION ---
def simulate_patient(patient_id, apply_glp1):
    # Seed per patient for reproducibility
    np.random.seed(patient_id * SEED_OFFSET) 
    torch.manual_seed(patient_id * SEED_OFFSET)
    
    current_age = 45.0
    context_tokens = [int(T_MALE), int(T_BMI_HIGH)]
    history_log = [f"START: Male, Age 45, BMI High"]
    
    total_costs, total_dalys, events = 0, 0, []

    for year in range(TIME_HORIZON):
        # Safety: Respect 48-token context window
        context_tokens = context_tokens[-48:] 

        # Intervention Step (The "Fudge")
        if apply_glp1 and str(context_tokens[-1]) == T_BMI_HIGH:
            if np.random.rand() < 0.8: # 80% Success rate jump
                context_tokens[-1] = int(T_BMI_MID)
                history_log.append(f"Age {int(current_age)}: GLP-1 Intervention (BMI -> Mid)")
            total_costs += GLP_YEARLY_COST 

        # Blackwell GPU Inference
        x = torch.tensor(context_tokens, dtype=torch.long, device=DEVICE)[None, ...]
        age_tensor = torch.tensor([current_age], dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            # Robust unpacking for Delphi return types
            out = model(x, age=age_tensor)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            probs = torch.softmax(logits[:, -1, :] / 1.0, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        
        context_tokens.append(next_token)
        t_str = str(next_token)
        
        # Log visible events
        label = labels_list[next_token]
        if label not in ["No event", "Padding"]:
            history_log.append(f"Age {int(current_age)}: {label}")

        # Health Economic Accrual
        if t_str in DISABILITY_WEIGHTS:
            total_dalys += DISABILITY_WEIGHTS[t_str]
            total_costs += DISEASE_COSTS.get(t_str, 0)
            events.append(label)
        
        if t_str == T_DEATH:
            total_dalys += 15.0 # YLL Placeholder
            break
        
        current_age += 1.0

    narrative = " -> ".join(history_log)
    
    # Console Output
    if PRINT_TRAJECTORIES and patient_id < NUM_PATIENTS:
        print(f"\n--- Patient {patient_id} Path ---")
        print(narrative)

    return {
        "ID": patient_id, 
        "Cost": total_costs, 
        "DALYs": total_dalys, 
        "Events": "|".join(events),
        "Full_History": narrative
    }

# --- 5. EXECUTION & SAVING ---
print(f"🚀 Running {'Intervention' if APPLY_INTERVENTION else 'Baseline'} on Blackwell GPU...")
results = [simulate_patient(i, APPLY_INTERVENTION) for i in range(NUM_PATIENTS)]
df = pd.DataFrame(results)

suffix = 'glp1' if APPLY_INTERVENTION else 'base'

# Save Summary CSV
df[['ID', 'Cost', 'DALYs', 'Events']].to_csv(f"delfino_results_{suffix}.csv", index=False)

# Save Trajectories File [NEW]
if SAVE_TRAJECTORY_FILES:
    with open(f"delfino_trajectories_{suffix}.txt", "w") as f:
        for idx, row in df.iterrows():
            f.write(f"Patient {row['ID']}:\n{row['Full_History']}\n{'-'*50}\n")
    print(f"📁 Full trajectories saved to delfino_trajectories_{suffix}.txt")

print("\n--- Summary Results ---")
print(df[['Cost', 'DALYs']].describe().loc[['mean', 'max']])