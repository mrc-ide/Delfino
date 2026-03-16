import os, torch, argparse, sys
import numpy as np
import pandas as pd
from model import Delphi, DelphiConfig
from tqdm import tqdm

# --- 1. ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument('--start_id', type=int, required=True)
parser.add_argument('--end_id', type=int, required=True)
parser.add_argument('--position', type=int, default=0)
parser.add_argument('--time_horizon', type=int, default=40)
parser.add_argument('--start_age', type=float, default=40.0)
parser.add_argument('--apply_intervention', action='store_true')
parser.add_argument('--seed_offset', type=int, default=42)
parser.add_argument('--logit_bias', type=float, default=0.0)
parser.add_argument('--pin_identity', type=str, default='true')
parser.add_argument('--remind_bmi', type=str, default='true')
args = parser.parse_args()

# --- 2. ENVIRONMENT SETUP ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join('data', 'ukb_simulated_data')
LIFE_EXPECTANCY = {40: 42, 45: 37, 50: 33, 55: 28, 60: 24, 65: 20, 70: 16, 75: 12, 80: 9, 85: 6, 90: 4}

with open(os.path.join(data_dir, 'labels.csv'), 'r') as f:
    labels_list = [line.strip() for line in f.readlines()]

checkpoint = torch.load('out-delfino-baseline/ckpt.pt', map_location=DEVICE)
VOCAB_SIZE = checkpoint['model_args']['vocab_size']

# UPFRONT OPTIMIZATION
GLOBAL_DW = np.zeros(VOCAB_SIZE)
GLOBAL_UTILITY = np.ones(VOCAB_SIZE)
GLOBAL_COSTS = np.zeros(VOCAB_SIZE)
IS_DISEASE = np.zeros(VOCAB_SIZE, dtype=bool)

disease_df = pd.read_csv('dummy_disease_params.csv')
for _, row in disease_df.iterrows():
    tid = int(row['TokenID'])
    GLOBAL_DW[tid] = row['DW']
    GLOBAL_UTILITY[tid] = row['Utility']
    GLOBAL_COSTS[tid] = row['Cost']
    IS_DISEASE[tid] = True

T_DEATH_ID = labels_list.index('Death')
T_BMI_HIGH_INT = labels_list.index('BMI_high')
T_BMI_MID = labels_list.index('BMI_mid')
TRACKED_CODES = dict(zip(disease_df['TokenID'], disease_df['Code']))

# --- 3. HARD-CODED BIAS & PREVALENCE DIAGNOSTICS ---
targets = ['E10', 'I50', 'I21', 'I63', 'N18']
massive_penalty = -100.0

# Initialize bias vector with the default logit_bias
logit_bias_vector = torch.full((VOCAB_SIZE,), args.logit_bias, device=DEVICE)

# Find IDs for targets and apply penalty
target_ids = {}
for code in targets:
    for i, label in enumerate(labels_list):
        if label.startswith(code):
            target_ids[code] = i
            logit_bias_vector[i] = massive_penalty
            break

# Diagnostics: Count background cases in the subgroup
all_patients_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)
subgroup_raw = all_patients_data[args.start_id*48 : args.end_id*48].reshape(-1, 48)

print(f"\n--- Subprocess Background Diagnostics ({args.start_id} to {args.end_id}) ---")
for code, tid in target_ids.items():
    count = np.any(subgroup_raw == tid, axis=1).sum()
    print(f"Background cases of {code}: {count}")

# Load model
model = Delphi(DelphiConfig(**checkpoint['model_args'])).to(DEVICE).eval()
model.load_state_dict(checkpoint['model'])

# --- 4. SIMULATION LOGIC ---
def simulate_patient(patient_id, apply_glp1):
    np.random.seed(patient_id * args.seed_offset)
    torch.manual_seed(patient_id * args.seed_offset)
    current_age = args.start_age
    
    # inc_record uses raw codes as keys (no "inc_" prefix)
    inc_record = {c: -1.0 for c in TRACKED_CODES.values()}
    total_costs, total_yld, total_yll, qalys_add, qalys_mult = 0.0, 0.0, 0.0, 0.0, 0.0
    active_dw, active_utilities, active_costs = [], [], []
    already_diagnosed = np.zeros(VOCAB_SIZE, dtype=bool)

    # Context Reconstruction
    raw = all_patients_data[patient_id*48 : (patient_id+1)*48].tolist()
    context = [t for t in raw if t < VOCAB_SIZE and labels_list[t] not in ["Padding", "No event"]]
    
    # Process background history: Record cases prevalent at baseline
    for t in context:
        if IS_DISEASE[t]:
            if not already_diagnosed[t]:
                active_dw.append(GLOBAL_DW[t])
                active_utilities.append(GLOBAL_UTILITY[t])
                active_costs.append(GLOBAL_COSTS[t])
                already_diagnosed[t] = True
            
            code = TRACKED_CODES[t]
            if inc_record[code] == -1.0:
                inc_record[code] = args.start_age

    if args.pin_identity.lower() == 'true':
        identity, history = context[:5], context[5:]
    else:
        identity, history = [], context

    has_lost_weight = False

    for year in range(args.time_horizon):
        total_yld += sum(active_dw)
        qalys_add += max(0, 1.0 - sum([1.0 - u for u in active_utilities]))
        
        u_step = 1.0
        for u in active_utilities: u_step *= u
        qalys_mult += u_step
        
        total_costs += sum(active_costs)

        if args.remind_bmi.lower() == 'true' and year % 5 == 0 and has_lost_weight:
            history.append(T_BMI_MID)

        max_hist = 48 - len(identity)
        full_context = identity + history[-max_hist:]
        
        if apply_glp1 and T_BMI_HIGH_INT in (identity + history):
            total_costs += 1200
            if not has_lost_weight and np.random.rand() < 0.8:
                identity = [t if t != T_BMI_HIGH_INT else T_BMI_MID for t in identity]
                history = [t if t != T_BMI_HIGH_INT else T_BMI_MID for t in history]
                has_lost_weight = True

        x = torch.tensor(full_context, dtype=torch.long, device=DEVICE)[None, ...]
        age_t = torch.tensor([current_age], dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            out = model(x, age=age_t)
            logits = (out[0] if isinstance(out, (list, tuple)) else out)[:, -1, :]
            
            if apply_glp1 and has_lost_weight:
                logits[0] += logit_bias_vector
                
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        
        if next_token == T_DEATH_ID:
            total_yll = LIFE_EXPECTANCY.get(int(current_age), 5.0)
            break
        
        if IS_DISEASE[next_token]:
            if not already_diagnosed[next_token]:
                active_dw.append(GLOBAL_DW[next_token])
                active_utilities.append(GLOBAL_UTILITY[next_token])
                active_costs.append(GLOBAL_COSTS[next_token])
                already_diagnosed[next_token] = True
            
            code = TRACKED_CODES[next_token]
            if inc_record[code] == -1.0:
                inc_record[code] = current_age

        history.append(next_token)
        current_age += 1.0

    return {"ID": patient_id, "Cost": total_costs, "YLD": total_yld, "YLL": total_yll, "DALYs": total_yld+total_yll, "QALYs_Add": qalys_add, "QALYs_Mult": qalys_mult, **inc_record}

# --- 5. EXECUTION ---
desc = f"{'GLP1' if args.apply_intervention else 'Base'} {args.start_id}-{args.end_id}"
results = [simulate_patient(i, args.apply_intervention) 
           for i in tqdm(range(args.start_id, args.end_id), position=args.position, leave=False, desc=desc)]

pd.DataFrame(results).to_csv(f"temp_{'glp1' if args.apply_intervention else 'base'}_{args.start_id}_{args.end_id}.csv", index=False)