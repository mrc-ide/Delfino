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

# UPFRONT OPTIMIZATION: Now with three arrays
GLOBAL_DW = np.zeros(VOCAB_SIZE)
GLOBAL_UTILITY = np.ones(VOCAB_SIZE) # Default utility is 1.0 (perfect health)
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
bias_indices = torch.tensor([int(k) for k in disease_df['TokenID'] if k != T_DEATH_ID], device=DEVICE)

model = Delphi(DelphiConfig(**checkpoint['model_args'])).to(DEVICE).eval()
model.load_state_dict(checkpoint['model'])
all_patients_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)

def simulate_patient(patient_id, apply_glp1):
    np.random.seed(patient_id * args.seed_offset)
    torch.manual_seed(patient_id * args.seed_offset)
    current_age = args.start_age
    
    inc_record = {f"inc_{c}": -1.0 for c in TRACKED_CODES.values()}
    total_costs, total_yld, total_yll, qalys_add, qalys_mult = 0.0, 0.0, 0.0, 0.0, 0.0
    active_dw, active_utilities, active_costs = [], [], []
    already_diagnosed = np.zeros(VOCAB_SIZE, dtype=bool)

    raw = all_patients_data[patient_id*48 : (patient_id+1)*48].tolist()
    context = [t for t in raw if t < VOCAB_SIZE and labels_list[t] not in ["Padding", "No event"]]
    
    for t in context:
        if IS_DISEASE[t]:
            inc_record[f"inc_{TRACKED_CODES[t]}"] = -99.0
            if not already_diagnosed[t]:
                active_dw.append(GLOBAL_DW[t])
                active_utilities.append(GLOBAL_UTILITY[t])
                active_costs.append(GLOBAL_COSTS[t])
                already_diagnosed[t] = True

    if args.pin_identity.lower() == 'true':
        identity, history = context[:5], context[5:]
    else:
        identity, history = [], context

    has_lost_weight = False

    for year in range(args.time_horizon):
        # 1. DALY Accumulation (Health Loss)
        total_yld += sum(active_dw)
        
        # 2. QALY Accumulation (Additive)
        # 1.0 - Sum of quality decrements
        qalys_add += max(0, 1.0 - sum([1.0 - u for u in active_utilities]))
        
        # 3. QALY Accumulation (Multiplicative Utility)
        # Standard NICE/HEOR approach
        u_step = 1.0
        for u in active_utilities: u_step *= u
        qalys_mult += u_step
        
        total_costs += sum(active_costs)

        if args.remind_bmi.lower() == 'true' and year % 5 == 0 and has_lost_weight:
            history.append(T_BMI_MID)

        max_hist = 48 - len(identity)
        full_context = identity + history[-max_hist:]
        
        if apply_glp1 and T_BMI_HIGH_INT in full_context:
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
            if apply_glp1 and has_lost_weight and args.logit_bias != 0:
                logits[0, bias_indices] += args.logit_bias
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        
        if next_token == T_DEATH_ID:
            total_yll = LIFE_EXPECTANCY.get(int(current_age), 5.0)
            break
        
        if IS_DISEASE[next_token] and not already_diagnosed[next_token]:
            active_dw.append(GLOBAL_DW[next_token])
            active_utilities.append(GLOBAL_UTILITY[next_token])
            active_costs.append(GLOBAL_COSTS[next_token])
            already_diagnosed[next_token] = True
            inc_record[f"inc_{TRACKED_CODES[next_token]}"] = current_age

        history.append(next_token)
        current_age +=