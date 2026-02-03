import os
import pickle
import torch
import numpy as np
import pandas as pd
import argparse
import sys
from model import Delphi, DelphiConfig
from tqdm import tqdm

# --- 1. CLI ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Delfino Simulation Engine')
parser.add_argument('--seed_offset', type=int, default=42)
parser.add_argument('--num_patients', type=int, default=100)
parser.add_argument('--time_horizon', type=int, default=20)
parser.add_argument('--start_age', type=float, default=45.0)
parser.add_argument('--apply_intervention', action='store_true')
parser.add_argument('--use_real_data', type=str, default='true')
parser.add_argument('--print_trajectories', action='store_true')
parser.add_argument('--num_to_print', type=int, default=10)
parser.add_argument('--save_trajectories', type=str, default='true')
args = parser.parse_args()

# Configuration mapping
SEED_OFFSET = args.seed_offset
NUM_PATIENTS = args.num_patients
TIME_HORIZON = args.time_horizon
START_AGE = args.start_age
APPLY_INTERVENTION = args.apply_intervention
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_REAL_DATA = args.use_real_data.lower() == 'true'
PRINT_TRAJECTORIES = args.print_trajectories
NUM_PATIENTS_TO_PRINT_TO_CONSOLE = args.num_to_print
SAVE_TRAJECTORY_FILES = args.save_trajectories.lower() == 'true'

# --- 2. DYNAMIC LABEL & DISEASE MAPPING ---
data_dir = os.path.join('data', 'ukb_simulated_data')
labels_path = os.path.join(data_dir, 'labels.csv')
with open(labels_path, 'r') as f:
    labels_list = [line.strip() for line in f.readlines()]

def get_id(label_name):
    try: return str(labels_list.index(label_name))
    except ValueError: return None

TRACKED_DISEASES = {
    get_id('E11 (non-insulin-dependent diabetes mellitus)'): 'diabetes',
    get_id('I50 (heart failure)'): 'heart_failure',
    get_id('I63 (cerebral infarction)'): 'stroke',
    get_id('J44 (other chronic obstructive pulmonary disease)'): 'copd',
    get_id('Death'): 'death'
}

DISABILITY_WEIGHTS = {k: v for k, v in zip(TRACKED_DISEASES.keys(), [0.05, 0.18, 0.23, 1.0, 0.22]) if k}
DISEASE_COSTS = {k: v for k, v in zip(list(TRACKED_DISEASES.keys())[:-1], [2500, 8000, 15000, 4500]) if k}
GLP_YEARLY_COST = 1200

# --- 3. LOAD MODEL & DATA ---
if USE_REAL_DATA:
    all_patients_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)

out_dir = 'out-delfino-baseline'
checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=DEVICE)
VOCAB_SIZE = checkpoint['model_args']['vocab_size']
model = Delphi(DelphiConfig(**checkpoint['model_args']))
model.load_state_dict(checkpoint['model'])
model.to(DEVICE).eval()

# --- 4. SIMULATOR CORE ---
def get_safe_label(token_id):
    if token_id < len(labels_list): return labels_list[token_id]
    return f"UNK_{token_id}"

def simulate_patient(patient_id, apply_glp1):
    # Set seeds for both numpy and torch
    np.random.seed(patient_id * SEED_OFFSET) 
    torch.manual_seed(patient_id * SEED_OFFSET)
    current_age = START_AGE
    
    incidence_record = {f"inc_{name}": -1.0 for name in TRACKED_DISEASES.values()}
    total_costs, total_dalys = 0, 0
    
    start_idx = patient_id * 48
    raw_tokens = all_patients_data[start_idx : start_idx + 48].tolist()
    sanitized = [t for t in raw_tokens if t < VOCAB_SIZE]
    context_tokens = [t for t in sanitized if get_safe_label(t) not in ["Padding", "No event"]]
    
    # Pre-existing check
    for t in context_tokens:
        if str(t) in TRACKED_DISEASES:
            incidence_record[f"inc_{TRACKED_DISEASES[str(t)]}"] = -99.0
    
    history_log = [f"START: Age {int(current_age)}"]

    for year in range(TIME_HORIZON):
        context_tokens = context_tokens[-48:] 
        str_context = [str(t) for t in context_tokens]
        T_BMI_HIGH = get_id('BMI_high')
        
        # RNG SYNC FIX: Consume the same random number in both runs
        intervention_roll = np.random.rand() 
        
        if str(T_BMI_HIGH) in str_context:
            if apply_glp1:
                total_costs += GLP_YEARLY_COST 
                if intervention_roll < 0.8: # Use the pre-rolled variable
                    last_idx = len(str_context) - 1 - str_context[::-1].index(str(T_BMI_HIGH))
                    context_tokens[last_idx] = int(get_id('BMI_mid'))
                    history_log.append(f"Age {int(current_age)}: GLP-1 Active")

        # Inference
        x = torch.tensor(context_tokens, dtype=torch.long, device=DEVICE)[None, ...]
        age_t = torch.tensor([current_age], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            out = model(x, age=age_t)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            # Sampling uses a different RNG (Torch) which is also seeded
            next_token = torch.multinomial(probs, 1).item()
        
        context_tokens.append(next_token)
        t_str = str(next_token)
        
        if t_str in TRACKED_DISEASES:
            key = f"inc_{TRACKED_DISEASES[t_str]}"
            if incidence_record[key] == -1.0:
                incidence_record[key] = current_age
            total_dalys += DISABILITY_WEIGHTS.get(t_str, 0)
            total_costs += DISEASE_COSTS.get(t_str, 0)
            history_log.append(f"Age {int(current_age)}: {get_safe_label(next_token)}")
            
        if t_str == get_id('Death'): break
        current_age += 1.0

    return {"ID": patient_id, "Cost": total_costs, "DALYs": total_dalys, **incidence_record, "History": " -> ".join(history_log)}

# --- 5. EXECUTION ---
suffix = 'glp1' if APPLY_INTERVENTION else 'base'
results = [simulate_patient(i, APPLY_INTERVENTION) for i in tqdm(range(NUM_PATIENTS), desc=f"Running {suffix.upper()}")]
df = pd.DataFrame(results)

try:
    df.drop(columns=['History']).to_csv(f"delfino_individual_{suffix}.csv", index=False)
except PermissionError:
    print(f"⚠️ Close Excel! Could not save delfino_individual_{suffix}.csv")

if SAVE_TRAJECTORY_FILES:
    with open(f"delfino_trajectories_{suffix}.txt", "w") as f:
        for _, row in df.iterrows():
            f.write(f"Patient {row['ID']}:\n{row['History']}\n{'-'*50}\n")