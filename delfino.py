import os, pickle, torch, argparse, sys
import numpy as np
import pandas as pd
from model import Delphi, DelphiConfig
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--num_patients', type=int, default=1000)
parser.add_argument('--time_horizon', type=int, default=30)
parser.add_argument('--start_age', type=float, default=45.0)
parser.add_argument('--apply_intervention', action='store_true')
parser.add_argument('--seed_offset', type=int, default=42)
parser.add_argument('--logit_bias', type=float, default=0.0) 
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join('data', 'ukb_simulated_data')

# --- 1. ROBUST LABEL LOADING ---
# We read lines directly to avoid pandas CSV parser errors with commas
with open(os.path.join(data_dir, 'labels.csv'), 'r') as f:
    labels_list = [line.strip() for line in f.readlines()]

def get_id(label_name):
    try: return str(labels_list.index(label_name))
    except ValueError: return None

# --- 2. DYNAMIC PARAMETER LOADING ---
disease_df = pd.read_csv('dummy_disease_params.csv')
TRACKED_DISEASES = dict(zip(disease_df['TokenID'].astype(str), disease_df['Code']))
DISABILITY_WEIGHTS = dict(zip(disease_df['TokenID'].astype(str), disease_df['Weight']))
DISEASE_COSTS = dict(zip(disease_df['TokenID'].astype(str), disease_df['Cost']))

# Vectorize bias indices for performance
T_DEATH_ID = get_id('Death')
# Pre-calculate indices to apply bias to (excludes Death)
bias_indices = torch.tensor([int(k) for k in TRACKED_DISEASES.keys() if k != T_DEATH_ID], device=DEVICE)

# --- 3. LOAD MODEL & DATA ---
checkpoint = torch.load('out-delfino-baseline/ckpt.pt', map_location=DEVICE)
model = Delphi(DelphiConfig(**checkpoint['model_args'])).to(DEVICE).eval()
model.load_state_dict(checkpoint['model'])
all_patients_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)
VOCAB_SIZE = checkpoint['model_args']['vocab_size']

def simulate_patient(patient_id, apply_glp1):
    np.random.seed(patient_id * args.seed_offset)
    torch.manual_seed(patient_id * args.seed_offset)
    current_age = args.start_age
    
    incidence = {f"inc_{c}": -1.0 for c in TRACKED_DISEASES.values()}
    total_costs, total_dalys = 0, 0
    
    start_idx = patient_id * 48
    raw = all_patients_data[start_idx : start_idx + 48].tolist()
    sanitized = [t for t in raw if t < VOCAB_SIZE]
    context = [t for t in sanitized if labels_list[t] not in ["Padding", "No event"]]
    
    T_BMI_HIGH = get_id('BMI_high')
    T_BMI_MID = int(get_id('BMI_mid'))

    # Pre-existing condition check
    for t in context:
        if str(t) in TRACKED_DISEASES:
            incidence[f"inc_{TRACKED_DISEASES[str(t)]}"] = -99.0

    identity, history = context[:5], context[5:]
    has_lost_weight = False

    for year in range(args.time_horizon):
        # 5-Year Reminder to keep weight loss in Transformer Attention window
        if year % 5 == 0 and has_lost_weight:
            history.append(T_BMI_MID)

        full_context = (identity + history)[-48:]
        intervention_roll = np.random.rand()
        
        if any(str(t) == T_BMI_HIGH for t in full_context) and apply_glp1:
            total_costs += 1200
            if intervention_roll < 0.8:
                full_context = [t if str(t) != T_BMI_HIGH else T_BMI_MID for t in full_context]
                identity = [t if str(t) != T_BMI_HIGH else T_BMI_MID for t in identity]
                has_lost_weight = True

        x = torch.tensor(full_context, dtype=torch.long, device=DEVICE)[None, ...]
        age_t = torch.tensor([current_age], dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            out = model(x, age=age_t)
            logits = (out[0] if isinstance(out, (list, tuple)) else out)[:, -1, :]
            
            # THE DIVERGENCE FIX: Apply bias if intervention is active and successful
            if apply_glp1 and has_lost_weight and args.logit_bias != 0:
                logits[0, bias_indices] += args.logit_bias # bias is usually negative
            
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        
        history.append(next_token)
        t_str = str(next_token)
        if t_str in TRACKED_DISEASES:
            key = f"inc_{TRACKED_DISEASES[t_str]}"
            if incidence[key] == -1.0:
                incidence[key] = current_age
            total_dalys += DISABILITY_WEIGHTS.get(t_str, 0)
            total_costs += DISEASE_COSTS.get(t_str, 0)
            
        if t_str == T_DEATH_ID: break
        current_age += 1.0

    return {"ID": patient_id, "Cost": total_costs, "DALYs": total_dalys, **incidence}

print(f"🚀 Running {'INTERVENTION' if args.apply_intervention else 'BASELINE'}...")
results = [simulate_patient(i, args.apply_intervention) for i in tqdm(range(args.num_patients))]
df = pd.DataFrame(results)
# Only save columns for diseases that actually occurred
cols = [c for c in df.columns if not c.startswith('inc_') or (df[c] > 0).any()]
df[cols].to_csv(f"delfino_individual_{'glp1' if args.apply_intervention else 'base'}.csv", index=False)