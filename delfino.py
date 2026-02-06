import os, pickle, torch, argparse, sys
import numpy as np
import pandas as pd
from model import Delphi, DelphiConfig
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--num_patients', type=int, default=1000)
parser.add_argument('--time_horizon', type=int, default=30)
parser.add_argument('--start_age', type=float, default=40.0)
parser.add_argument('--apply_intervention', action='store_true')
parser.add_argument('--seed_offset', type=int, default=42)
parser.add_argument('--logit_bias', type=float, default=-2.0)
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join('data', 'ukb_simulated_data')

# Simplified Life Expectancy (Age: Expected additional years)
# Source: Simplified ONS Life Tables
LIFE_EXPECTANCY = {40: 42, 45: 37, 50: 33, 55: 28, 60: 24, 65: 20, 70: 16, 75: 12, 80: 9, 85: 6, 90: 4}

with open(os.path.join(data_dir, 'labels.csv'), 'r') as f:
    labels_list = [line.strip() for line in f.readlines()]

def get_id(label_name):
    try: return str(labels_list.index(label_name))
    except ValueError: return None

disease_df = pd.read_csv('dummy_disease_params.csv')
TRACKED_DISEASES = dict(zip(disease_df['TokenID'].astype(str), disease_df['Code']))
DISABILITY_WEIGHTS = dict(zip(disease_df['TokenID'].astype(str), disease_df['Weight']))
DISEASE_COSTS = dict(zip(disease_df['TokenID'].astype(str), disease_df['Cost']))

T_DEATH_ID = get_id('Death')
T_BMI_HIGH, T_BMI_MID = str(get_id('BMI_high')), int(get_id('BMI_mid'))
bias_indices = torch.tensor([int(k) for k in TRACKED_DISEASES.keys() if k != T_DEATH_ID], device=DEVICE)

checkpoint = torch.load('out-delfino-baseline/ckpt.pt', map_location=DEVICE)
model = Delphi(DelphiConfig(**checkpoint['model_args'])).to(DEVICE).eval()
model.load_state_dict(checkpoint['model'])
all_patients_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)
VOCAB_SIZE = checkpoint['model_args']['vocab_size']

def simulate_patient(patient_id, apply_glp1):
    np.random.seed(patient_id * args.seed_offset)
    torch.manual_seed(patient_id * args.seed_offset)
    current_age = args.start_age
    
    # State tracking
    incidence = {f"inc_{c}": -1.0 for c in TRACKED_DISEASES.values()}
    total_costs = 0.0
    total_yld = 0.0
    total_yll = 0.0
    total_qalys = 0.0
    active_conditions = set()

    raw = all_patients_data[patient_id*48 : (patient_id+1)*48].tolist()
    context = [t for t in raw if t < VOCAB_SIZE and labels_list[t] not in ["Padding", "No event"]]
    
    for t in context:
        t_str = str(t)
        if t_str in TRACKED_DISEASES: 
            incidence[f"inc_{TRACKED_DISEASES[t_str]}"] = -99.0
            active_conditions.add(t_str)

    identity, history = context[:5], context[5:]
    has_lost_weight = False

    for year in range(args.time_horizon):
        # Accumulate yearly disability and quality
        year_weight = sum(DISABILITY_WEIGHTS.get(c, 0) for c in active_conditions)
        total_yld += year_weight
        total_qalys += max(0, 1.0 - year_weight)
        total_costs += sum(DISEASE_COSTS.get(c, 0) for c in active_conditions)

        if year % 5 == 0 and has_lost_weight: history.append(T_BMI_MID)
        full_context = (identity + history)[-48:]
        
        if apply_glp1:
            if any(str(t) == T_BMI_HIGH for t in full_context):
                total_costs += 1200 # Intervention cost
                if np.random.rand() < 0.8:
                    identity = [t if str(t) != T_BMI_HIGH else T_BMI_MID for t in identity]
                    has_lost_weight = True

        x = torch.tensor((identity + history)[-48:], dtype=torch.long, device=DEVICE)[None, ...]
        age_t = torch.tensor([current_age], dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            out = model(x, age=age_t)
            logits = (out[0] if isinstance(out, (list, tuple)) else out)[:, -1, :]
            if apply_glp1 and has_lost_weight and args.logit_bias != 0:
                logits[0, bias_indices] += args.logit_bias
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        
        history.append(next_token)
        t_str = str(next_token)
        
        if t_str == T_DEATH_ID:
            # Calculate Years of Life Lost
            expected = LIFE_EXPECTANCY.get(int(current_age), LIFE_EXPECTANCY[max(LIFE_EXPECTANCY.keys())])
            total_yll = expected
            break
            
        if t_str in TRACKED_DISEASES:
            active_conditions.add(t_str)
            key = f"inc_{TRACKED_DISEASES[t_str]}"
            if incidence[key] == -1.0: incidence[key] = current_age
        
        current_age += 1.0

    return {
        "ID": patient_id, "Cost": total_costs, 
        "YLD": total_yld, "YLL": total_yll, 
        "DALYs": total_yld + total_yll, "QALYs": total_qalys,
        **incidence
    }

results = [simulate_patient(i, args.apply_intervention) for i in tqdm(range(args.num_patients))]
df = pd.DataFrame(results)
cols = [c for c in df.columns if not c.startswith('inc_') or (df[c] > 0).any()]
df[cols].to_csv(f"delfino_individual_{'glp1' if args.apply_intervention else 'base'}.csv", index=False)