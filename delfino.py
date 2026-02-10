import os, pickle, torch, argparse, sys
import numpy as np
import pandas as pd
from model import Delphi, DelphiConfig
from tqdm import tqdm

# --- 1. CLI & CONFIG ---
parser = argparse.ArgumentParser()
parser.add_argument('--num_patients', type=int, default=1000)
parser.add_argument('--time_horizon', type=int, default=30)
parser.add_argument('--start_age', type=float, default=40.0)
parser.add_argument('--apply_intervention', action='store_true')
parser.add_argument('--seed_offset', type=int, default=42)
parser.add_argument('--logit_bias', type=float, default=0.0)
# Feature Toggles
parser.add_argument('--pin_identity', type=str, default='true')
parser.add_argument('--remind_bmi', type=str, default='true')
args = parser.parse_args()

def str_to_bool(v): return v.lower() == 'true'
PIN_IDENTITY = str_to_bool(args.pin_identity)
REMIND_BMI = str_to_bool(args.remind_bmi)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join('data', 'ukb_simulated_data')

# Simplified Life Expectancy Table (Age: Expected additional years)
# Based on ONS-style actuarial data
LIFE_EXPECTANCY = {
    40: 42, 45: 37, 50: 33, 55: 28, 60: 24, 
    65: 20, 70: 16, 75: 12, 80: 9, 85: 6, 90: 4
}

# --- 2. ROBUST DATA LOADING ---
with open(os.path.join(data_dir, 'labels.csv'), 'r') as f:
    labels_list = [line.strip() for line in f.readlines()]

checkpoint = torch.load('out-delfino-baseline/ckpt.pt', map_location=DEVICE)
VOCAB_SIZE = checkpoint['model_args']['vocab_size']

# UPFRONT OPTIMIZATION: Flat Arrays for O(1) Lookup (C++ Array style)
GLOBAL_WEIGHTS = np.zeros(VOCAB_SIZE)
GLOBAL_COSTS = np.zeros(VOCAB_SIZE)
IS_DISEASE = np.zeros(VOCAB_SIZE, dtype=bool)

disease_df = pd.read_csv('dummy_disease_params.csv')
for _, row in disease_df.iterrows():
    tid = int(row['TokenID'])
    GLOBAL_WEIGHTS[tid] = row['Weight']
    GLOBAL_COSTS[tid] = row['Cost']
    IS_DISEASE[tid] = True

# Key Token Lookups
def get_id(label_name):
    try: return labels_list.index(label_name)
    except ValueError: return -1

T_DEATH_ID = get_id('Death')
T_BMI_HIGH_STR = str(get_id('BMI_high'))
T_BMI_HIGH_INT = get_id('BMI_high')
T_BMI_MID = get_id('BMI_mid')
TRACKED_CODES = dict(zip(disease_df['TokenID'], disease_df['Code']))

# Pre-calc indices for Logit Bias (Vectorized)
bias_indices = torch.tensor([int(k) for k in disease_df['TokenID'] if k != T_DEATH_ID], device=DEVICE)

# Load Model
model = Delphi(DelphiConfig(**checkpoint['model_args'])).to(DEVICE).eval()
model.load_state_dict(checkpoint['model'])
all_patients_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)

def simulate_patient(patient_id, apply_glp1):
    np.random.seed(patient_id * args.seed_offset)
    torch.manual_seed(patient_id * args.seed_offset)
    current_age = args.start_age
    
    inc_record = {f"inc_{c}": -1.0 for c in TRACKED_CODES.values()}
    total_costs = 0.0
    total_yld, total_yll = 0.0, 0.0
    qalys_add, qalys_mult = 0.0, 0.0
    
    active_weights = []
    active_costs = []
    already_diagnosed = np.zeros(VOCAB_SIZE, dtype=bool)

    raw = all_patients_data[patient_id*48 : (patient_id+1)*48].tolist()
    context = [t for t in raw if t < VOCAB_SIZE and labels_list[t] not in ["Padding", "No event"]]
    
    # Process Baseline conditions
    for t in context:
        if IS_DISEASE[t]:
            inc_record[f"inc_{TRACKED_CODES[t]}"] = -99.0
            if not already_diagnosed[t]:
                active_weights.append(GLOBAL_WEIGHTS[t])
                active_costs.append(GLOBAL_COSTS[t])
                already_diagnosed[t] = True

    # OPTIONAL FEATURE: Identity Pinning
    if PIN_IDENTITY:
        identity, history = context[:5], context[5:]
    else:
        identity, history = [], context

    has_lost_weight = False

    for year in range(args.time_horizon):
        # --- 1. HEALTH ECONOMICS (Additive vs Multiplicative) ---
        # Additive (DW Sum)
        year_disability = sum(active_weights)
        total_yld += year_disability
        qalys_add += max(0, 1.0 - year_disability)
        
        # Multiplicative (Utility Product)
        utility = 1.0
        for w in active_weights:
            utility *= (1.0 - w)
        qalys_mult += utility
        
        total_costs += sum(active_costs)

        # --- 2. INTERVENTION LOGIC ---
        # OPTIONAL FEATURE: BMI Reminder
        if REMIND_BMI and year % 5 == 0 and has_lost_weight:
            history.append(T_BMI_MID)

        full_context = (identity + history)[-48:]
        
        if apply_glp1 and any(str(t) == T_BMI_HIGH_STR for t in full_context):
            total_costs += 1200
            if not has_lost_weight and np.random.rand() < 0.8:
                # Update Identity/History to reflect weight loss
                identity = [t if t != T_BMI_HIGH_INT else T_BMI_MID for t in identity]
                history = [t if t != T_BMI_HIGH_INT else T_BMI_MID for t in history]
                has_lost_weight = True

        # --- 3. INFERENCE ---
        x = torch.tensor(full_context, dtype=torch.long, device=DEVICE)[None, ...]
        age_t = torch.tensor([current_age], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            out = model(x, age=age_t)
            logits = (out[0] if isinstance(out, (list, tuple)) else out)[:, -1, :]
            
            if apply_glp1 and has_lost_weight and args.logit_bias != 0:
                logits[0, bias_indices] += args.logit_bias
            
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        
        if next_token == T_DEATH_ID:
            total_yll = LIFE_EXPECTANCY.get(int(current_age), 5.0)
            break
        
        # Update Diagnosis State
        if IS_DISEASE[next_token] and not already_diagnosed[next_token]:
            active_weights.append(GLOBAL_WEIGHTS[next_token])
            active_costs.append(GLOBAL_COSTS[next_token])
            already_diagnosed[next_token] = True
            inc_record[f"inc_{TRACKED_CODES[next_token]}"] = current_age

        history.append(next_token)
        current_age += 1.0

    return {
        "ID": patient_id, "Cost": total_costs, 
        "YLD": total_yld, "YLL": total_yll,
        "DALYs": total_yld + total_yll, 
        "QALYs_Add": qalys_add, "QALYs_Mult": qalys_mult,
        **inc_record
    }

print(f"🧬 Delfino Engine v9.2: Pinning={PIN_IDENTITY}, Reminder={REMIND_BMI}")
results = [simulate_patient(i, args.apply_intervention) for i in tqdm(range(args.num_patients))]
df = pd.DataFrame(results)
cols = [c for c in df.columns if not c.startswith('inc_') or (df[c] > 0).any()]
df[cols].to_csv(f"delfino_individual_{'glp1' if args.apply_intervention else 'base'}.csv", index=False)