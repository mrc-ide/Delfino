import os, torch, argparse, sys
import numpy as np
import pandas as pd
from model import Delphi, DelphiConfig
from tqdm import tqdm

# --- 1. ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--end_id', type=int, default=50)
parser.add_argument('--time_horizon', type=int, default=20)
parser.add_argument('--max_new_tokens', type=int, default=100)
parser.add_argument('--start_age', type=float, default=40.0)
parser.add_argument('--apply_intervention', action='store_true', default=False)
parser.add_argument('--seed_offset', type=int, default=42)
parser.add_argument('--logit_bias', type=float, default=0.0)
parser.add_argument('--pin_identity', type=str, default='true')
parser.add_argument('--output_trajectories', type=str, default='true')
args = parser.parse_args()

# --- 2. ENVIRONMENT SETUP ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DAYS_PER_YEAR = 365.25
T_DEATH_ID = 1269 
data_dir = os.path.join('data', 'ukb_simulated_data')

with open(os.path.join(data_dir, 'labels.csv'), 'r') as f:
    labels_list = [line.strip() for line in f.readlines()]

checkpoint = torch.load('out-delfino-baseline/ckpt.pt', map_location=DEVICE)
VOCAB_SIZE = checkpoint['model_args']['vocab_size']

# HEOR Initialization
GLOBAL_DW, GLOBAL_UTILITY, GLOBAL_COSTS = np.zeros(VOCAB_SIZE), np.ones(VOCAB_SIZE), np.zeros(VOCAB_SIZE)
IS_DISEASE = np.zeros(VOCAB_SIZE, dtype=bool)
disease_df = pd.read_csv('dummy_disease_params.csv')
for _, row in disease_df.iterrows():
    tid = int(row['TokenID'])
    GLOBAL_DW[tid], GLOBAL_UTILITY[tid], GLOBAL_COSTS[tid] = row['DW'], row['Utility'], row['Cost']
    IS_DISEASE[tid] = True

T_BMI_HIGH_INT = labels_list.index('BMI_high')
T_BMI_MID = labels_list.index('BMI_mid')
TRACKED_CODES = dict(zip(disease_df['TokenID'], disease_df['Code']))

# --- 3. THE OFFICIAL DATA LOADER ---
# A. Load 3-column uint32 data
train_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint32).reshape(-1, 3)

# B. EXACT get_p2i implementation from official utils.py
def get_p2i(data):
    px = data[:, 0].astype('int')
    p2i = []
    j = 0
    q = px[0]
    for i, p in enumerate(px):
        if p != q:
            p2i.append([j, i - j])
            q = p
            j = i
        if i == len(px) - 1:
            p2i.append([j, i - j + 1])
    return np.array(p2i)

train_p2i = get_p2i(train_data)

# C. Logit Bias Setup
targets = ['E10', 'I50', 'I21', 'I63', 'N18']
logit_bias_vector = torch.full((VOCAB_SIZE,), args.logit_bias, device=DEVICE)
target_ids = {code: i for code in targets for i, label in enumerate(labels_list) if label.startswith(code)}
for tid in target_ids.values():
    logit_bias_vector[tid] = -100.0 

# D. Diagnostics using the official mapping
print(f"\n--- Subprocess Background Diagnostics ({args.start_id} to {args.end_id}) ---")
for code, tid in target_ids.items():
    count = 0
    # In the 3-column format, the p2i index IS the patient ID
    for pid in range(args.start_id, min(args.end_id, len(train_p2i))):
        start, length = train_p2i[pid]
        p_slice = train_data[start:start+length, 2]
        if tid in p_slice:
            count += 1
    print(f"Background cases of {code}: {count}")

model = Delphi(DelphiConfig(**checkpoint['model_args'])).to(DEVICE).eval()
model.load_state_dict(checkpoint['model'])

# --- 4. SIMULATION LOGIC ---
def simulate_patient(patient_id, apply_glp1):
    torch.manual_seed(patient_id * args.seed_offset)
    np.random.seed(patient_id * args.seed_offset)
    
    # Use official p2i mapping to get this specific patient
    if patient_id < len(train_p2i):
        start, length = train_p2i[patient_id]
        p_history = train_data[start:start+length]
        input_tokens = p_history[:, 2].tolist()
        input_ages = p_history[:, 1].astype(float).tolist()
    else:
        # Fallback if ID is out of bounds
        input_tokens, input_ages = [2], [args.start_age * DAYS_PER_YEAR]

    inc_record = {c: -1.0 for c in TRACKED_CODES.values()}
    active_dw, active_utilities, active_costs = [], [], []
    already_diagnosed = np.zeros(VOCAB_SIZE, dtype=bool)
    total_costs, total_yld, qalys_mult = 0.0, 0.0, 0.0

    input_strings = []
    for t, age_d in zip(input_tokens, input_ages):
        input_strings.append(f"{age_d/DAYS_PER_YEAR:2.1f}: {labels_list[t]}")
        if IS_DISEASE[t]:
            if not already_diagnosed[t]:
                active_dw.append(GLOBAL_DW[t]); active_utilities.append(GLOBAL_UTILITY[t]); active_costs.append(GLOBAL_COSTS[t])
                already_diagnosed[t] = True
            code = TRACKED_CODES.get(t)
            if code and inc_record[code] == -1.0: inc_record[code] = -99.0

    curr_tokens, curr_ages = input_tokens[:], input_ages[:]
    generated_strings = []
    max_age_days = (args.start_age + args.time_horizon) * DAYS_PER_YEAR
    last_age_days = curr_ages[-1]
    has_lost_weight = False

    for _ in range(args.max_new_tokens):
        x = torch.tensor([curr_tokens[-48:]], dtype=torch.long, device=DEVICE)
        a = torch.tensor([curr_ages[-48:]], dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            out = model(x, age=a)
            logits = out[0][0, -1, :]
            
            # Safe age prediction extraction from Delphi head
            if isinstance(out, (list, tuple)) and out[1] is not None:
                next_age_days = out[1][0, -1].item()
            else:
                next_age_days = last_age_days + DAYS_PER_YEAR

            if apply_glp1 and has_lost_weight:
                logits += logit_bias_vector
            
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()

        if next_token == T_DEATH_ID or next_age_days > max_age_days:
            if next_token == T_DEATH_ID:
                generated_strings.append(f"{next_age_days/DAYS_PER_YEAR:2.1f}: Death")
            break

        dt_yr = max(0, (next_age_days - last_age_days) / DAYS_PER_YEAR)
        total_yld += sum(active_dw) * dt_yr
        u_step = 1.0
        for u in active_utilities: u_step *= u
        qalys_mult += u_step * dt_yr
        total_costs += sum(active_costs) * dt_yr
        if apply_glp1: total_costs += 1200 * dt_yr

        if apply_glp1 and T_BMI_HIGH_INT in curr_tokens and not has_lost_weight:
            if np.random.rand() < 0.8:
                curr_tokens = [t if t != T_BMI_HIGH_INT else T_BMI_MID for t in curr_tokens]
                has_lost_weight = True

        if IS_DISEASE[next_token]:
            if not already_diagnosed[next_token]:
                active_dw.append(GLOBAL_DW[next_token]); active_utilities.append(GLOBAL_UTILITY[next_token]); active_costs.append(GLOBAL_COSTS[next_token])
                already_diagnosed[next_token] = True
            code = TRACKED_CODES.get(next_token)
            if code and inc_record[code] == -1.0: inc_record[code] = next_age_days / DAYS_PER_YEAR

        curr_tokens.append(next_token); curr_ages.append(next_age_days)
        generated_strings.append(f"{next_age_days/DAYS_PER_YEAR:2.1f}: {labels_list[next_token]}")
        last_age_days = next_age_days

    traj_str = "Input trajectory:\n" + "\n".join(input_strings) + "\n=====================\nGenerated trajectory:\n" + "\n".join(generated_strings)
    return {"metrics": {"ID": patient_id, "Cost": total_costs, "DALYs": total_yld, "QALYs": qalys_mult, **inc_record}, "trajectory": {"ID": str(patient_id), "Trajectory": traj_str}}

# --- 5. EXECUTION ---
results = [simulate_patient(i, args.apply_intervention) for i in tqdm(range(args.start_id, args.end_id), desc="Simulating")]
pd.DataFrame([r['metrics'] for r in results]).to_csv(f"temp_{'glp1' if args.apply_intervention else 'base'}_{args.start_id}_{args.end_id}_results.csv", index=False)
if args.output_trajectories.lower() == 'true':
    pd.DataFrame([r['trajectory'] for r in results]).set_index('ID').T.to_csv(f"temp_{'glp1' if args.apply_intervention else 'base'}_{args.start_id}_{args.end_id}_trajectories.csv", index=False)