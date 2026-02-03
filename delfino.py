import os
import pickle
import torch
import numpy as np
import pandas as pd
from model import Delphi, DelphiConfig
from tqdm import tqdm

# --- 1. CONFIGURATION & SEEDING ---
SEED_OFFSET = 42          
NUM_PATIENTS = 100        
TIME_HORIZON = 20         
START_AGE = 45.0          
APPLY_INTERVENTION = False 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- DATA & OUTPUT OPTIONS ---
USE_REAL_DATA = True      
PRINT_TRAJECTORIES = True 
NUM_PATIENTS_TO_PRINT_TO_CONSOLE = 10 
SAVE_TRAJECTORY_FILES = True  

# --- 2. DYNAMIC LABEL & DATA LOADING ---
data_dir = os.path.join('data', 'ukb_simulated_data')
labels_path = os.path.join(data_dir, 'labels.csv')

with open(labels_path, 'r') as f:
    labels_list = [line.strip() for line in f.readlines()]

def get_id(label_name):
    try: return str(labels_list.index(label_name))
    except ValueError: return None

T_BMI_HIGH = get_id('BMI_high')
T_BMI_MID  = get_id('BMI_mid')
T_DIABETES = get_id('E11 (non-insulin-dependent diabetes mellitus)')
T_HF       = get_id('I50 (heart failure)')
T_STROKE   = get_id('I63 (cerebral infarction)')
T_DEATH    = get_id('Death')

DISABILITY_WEIGHTS = {T_DIABETES: 0.05, T_HF: 0.18, T_STROKE: 0.23, T_DEATH: 1.0}
DISEASE_COSTS = {T_DIABETES: 2500, T_HF: 8000, T_STROKE: 15000}
GLP_YEARLY_COST = 1200

if USE_REAL_DATA:
    all_patients_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)

# --- 3. MODEL INITIALIZATION ---
out_dir = 'out-delfino-baseline'
checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=DEVICE)
VOCAB_SIZE = checkpoint['model_args']['vocab_size']
model = Delphi(DelphiConfig(**checkpoint['model_args']))
model.load_state_dict(checkpoint['model'])
model.to(DEVICE).eval()

# --- 4. SIMULATOR CORE ---
def get_safe_label(token_id):
    if token_id < len(labels_list):
        return labels_list[token_id]
    return f"UnknownToken_{token_id}"

def simulate_patient(patient_id, apply_glp1):
    np.random.seed(patient_id * SEED_OFFSET) 
    torch.manual_seed(patient_id * SEED_OFFSET)
    current_age = START_AGE
    
    if USE_REAL_DATA:
        start_idx = patient_id * 48
        raw_tokens = all_patients_data[start_idx : start_idx + 48].tolist()
        sanitized = [t for t in raw_tokens if t < VOCAB_SIZE]
        context_tokens = [t for t in sanitized if get_safe_label(t) not in ["Padding", "No event"]]
        
        # TEST OVERRIDE: Force Patient 99 to be healthy (verify eligibility logic)
        if patient_id == 99:
            context_tokens = [t if str(t) != T_BMI_HIGH else int(T_BMI_MID) for t in context_tokens]
            
        history_log = [f"START: Age {int(current_age)}, {', '.join([get_safe_label(t) for t in context_tokens])}"]
    else:
        context_tokens = [int(get_id('Male')), int(T_BMI_HIGH)]
        history_log = [f"START: Age {int(current_age)}, Male, BMI High"]

    total_costs, total_dalys, events = 0, 0, []

    for year in range(TIME_HORIZON):
        context_tokens = context_tokens[-48:] 

        # INTERVENTION: Identify most recent BMI_high for eligibility
        # We find the LAST index to ensure we intervene on 'current' BMI, not history
        try:
            # We convert to string list for easier index matching
            str_context = [str(t) for t in context_tokens]
            # rindex finds the last occurrence of the value
            last_bmi_idx = len(str_context) - 1 - str_context[::-1].index(T_BMI_HIGH)
            is_eligible = True
        except ValueError:
            is_eligible = False
        
        if apply_glp1 and is_eligible:
            total_costs += GLP_YEARLY_COST 
            # 80% Success: Change ONLY the most recent BMI token
            if np.random.rand() < 0.8:
                context_tokens[last_bmi_idx] = int(T_BMI_MID)
                history_log.append(f"Age {int(current_age)}: GLP-1 Success (Current BMI -> Mid)")

        # Blackwell GPU Inference
        x = torch.tensor(context_tokens, dtype=torch.long, device=DEVICE)[None, ...]
        age_t = torch.tensor([current_age], dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            out = model(x, age=age_t)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            probs = torch.softmax(logits[:, -1, :] / 1.0, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        
        context_tokens.append(next_token)
        label = get_safe_label(next_token)
        if label not in ["No event", "Padding"]:
            history_log.append(f"Age {int(current_age)}: {label}")

        t_str = str(next_token)
        if t_str in DISABILITY_WEIGHTS:
            total_dalys += DISABILITY_WEIGHTS[t_str]
            total_costs += DISEASE_COSTS.get(t_str, 0)
            events.append(label)
        
        if t_str == T_DEATH:
            total_dalys += 15.0 
            break
        current_age += 1.0

    narrative = " -> ".join(history_log)
    if PRINT_TRAJECTORIES and patient_id < NUM_PATIENTS_TO_PRINT_TO_CONSOLE:
        print(f"\n--- Patient {patient_id} ---\n{narrative}")

    return {"ID": patient_id, "Cost": total_costs, "DALYs": total_dalys, "Events": "|".join(events), "Full_History": narrative}

# --- 5. EXECUTION ---
print(f"🚀 Running {'Intervention' if APPLY_INTERVENTION else 'Baseline'} on Blackwell GPU...")
results = [simulate_patient(i, APPLY_INTERVENTION) for i in tqdm(range(NUM_PATIENTS))]

df = pd.DataFrame(results)
suffix = 'glp1' if APPLY_INTERVENTION else 'base'
df[['ID', 'Cost', 'DALYs', 'Events']].to_csv(f"delfino_results_{suffix}.csv", index=False)

p99 = df[df['ID'] == 99].iloc[0]
print(f"\n🧪 Verification: Patient 99 (Forced Healthy) Cost: ${p99['Cost']:.2f} (Target: 0.0)")

print("\n--- Summary Statistics ---")
print(df[['Cost', 'DALYs']].describe().loc[['mean', 'max']])