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
PRINT_TRAJECTORIES = True # Restored variable to fix NameError
NUM_PATIENTS_TO_PRINT_TO_CONSOLE = 10 
SAVE_TRAJECTORY_FILES = True  

# --- 2. DYNAMIC LABEL & DISEASE MAPPING ---
data_dir = os.path.join('data', 'ukb_simulated_data')
labels_path = os.path.join(data_dir, 'labels.csv')

with open(labels_path, 'r') as f:
    labels_list = [line.strip() for line in f.readlines()]

def get_id(label_name):
    try: return str(labels_list.index(label_name))
    except ValueError: return None

# Tracked diseases for the new Individual/Population output tables
# Mapping {TokenID: ShortName}
TRACKED_DISEASES = {
    get_id('E11 (non-insulin-dependent diabetes mellitus)'): 'diabetes',
    get_id('I50 (heart failure)'): 'heart_failure',
    get_id('I63 (cerebral infarction)'): 'stroke',
    get_id('J44 (other chronic obstructive pulmonary disease)'): 'copd',
    get_id('J45 (asthma)'): 'asthma',
    get_id('Death'): 'death'
}

# Health Econ weights - aligned with keys in TRACKED_DISEASES
DISABILITY_WEIGHTS = {k: v for k, v in zip(TRACKED_DISEASES.keys(), [0.05, 0.18, 0.23, 1.0, 0.22, 0.05]) if k}
DISEASE_COSTS = {k: v for k, v in zip(list(TRACKED_DISEASES.keys())[:-1], [2500, 8000, 15000, 4500, 1200]) if k}
GLP_YEARLY_COST = 1200

# Load binary training data via memory mapping
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
    
    # INDIVIDUAL-LEVEL TRACKING: Store Age of first incidence
    incidence_record = {f"inc_{name}": None for name in TRACKED_DISEASES.values()}
    total_costs, total_dalys = 0, 0
    
    # Initialize from Real Data
    start_idx = patient_id * 48
    raw_tokens = all_patients_data[start_idx : start_idx + 48].tolist()
    sanitized = [t for t in raw_tokens if t < VOCAB_SIZE]
    context_tokens = [t for t in sanitized if get_safe_label(t) not in ["Padding", "No event"]]
    
    # Mark Pre-existing conditions (found in the dataset at Age 45)
    for t in context_tokens:
        t_str = str(t)
        if t_str in TRACKED_DISEASES:
            incidence_record[f"inc_{TRACKED_DISEASES[t_str]}"] = "Pre-existing"

    history_log = [f"START: Age {int(current_age)}, {', '.join([get_safe_label(t) for t in context_tokens])}"]

    for year in range(TIME_HORIZON):
        context_tokens = context_tokens[-48:] 

        # INTERVENTION: Identify most recent BMI_high for eligibility
        str_context = [str(t) for t in context_tokens]
        T_BMI_HIGH = get_id('BMI_high')
        if str(T_BMI_HIGH) in str_context:
            if apply_glp1:
                total_costs += GLP_YEARLY_COST 
                if np.random.rand() < 0.8: # 80% Success Rate
                    last_idx = len(str_context) - 1 - str_context[::-1].index(str(T_BMI_HIGH))
                    context_tokens[last_idx] = int(get_id('BMI_mid'))
                    history_log.append(f"Age {int(current_age)}: GLP-1 (BMI High -> Mid)")

        # Blackwell GPU Inference
        x = torch.tensor(context_tokens, dtype=torch.long, device=DEVICE)[None, ...]
        age_t = torch.tensor([current_age], dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            out = model(x, age=age_t)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            next_token = torch.multinomial(torch.softmax(logits[:, -1, :], dim=-1), 1).item()
        
        context_tokens.append(next_token)
        t_str = str(next_token)
        label = get_safe_label(next_token)

        # UPDATE INCIDENCE & ECONOMICS
        if t_str in TRACKED_DISEASES:
            key = f"inc_{TRACKED_DISEASES[t_str]}"
            # Record first diagnosis during simulation only if not already recorded
            if incidence_record[key] is None:
                incidence_record[key] = int(current_age)
            
            total_dalys += DISABILITY_WEIGHTS.get(t_str, 0)
            total_costs += DISEASE_COSTS.get(t_str, 0)
            if label not in ["No event", "Padding"]:
                history_log.append(f"Age {int(current_age)}: {label}")
        
        if t_str == get_id('Death'): break
        current_age += 1.0

    narrative = " -> ".join(history_log)
    if PRINT_TRAJECTORIES and patient_id < NUM_PATIENTS_TO_PRINT_TO_CONSOLE:
        print(f"\n--- Patient {patient_id} ---\n{narrative}")

    # Aggregated record for this patient
    res = {"ID": patient_id, "Cost": total_costs, "DALYs": total_dalys}
    res.update(incidence_record)
    res["Full_History"] = narrative
    return res

# --- 5. EXECUTION & AGGREGATION ---
print(f"🚀 Running {'Intervention' if APPLY_INTERVENTION else 'Baseline'}...")
results = [simulate_patient(i, APPLY_INTERVENTION) for i in tqdm(range(NUM_PATIENTS))]
df_indiv = pd.DataFrame(results)

# B. POPULATION LEVEL: Aggregate yearly incidence counts
pop_incidence = []
for year in range(int(START_AGE), int(START_AGE + TIME_HORIZON)):
    year_stats = {"Year_Age": year}
    for disease in TRACKED_DISEASES.values():
        # Count patients who were diagnosed specifically at this age
        count = (df_indiv[f"inc_{disease}"] == year).sum()
        year_stats[disease] = count
    pop_incidence.append(year_stats)

df_pop = pd.DataFrame(pop_incidence)

# SAVE OUTPUTS
suffix = 'glp1' if APPLY_INTERVENTION else 'base'
df_indiv.drop(columns=['Full_History']).to_csv(f"delfino_individual_{suffix}.csv", index=False)
df_pop.to_csv(f"delfino_population_{suffix}.csv", index=False)

if SAVE_TRAJECTORY_FILES:
    with open(f"delfino_trajectories_{suffix}.txt", "w") as f:
        for idx, row in df_indiv.iterrows():
            f.write(f"Patient {row['ID']}:\n{row['Full_History']}\n{'-'*50}\n")

print(f"\n✅ Population stats saved to delfino_population_{suffix}.csv")
print(f"\n--- Population Incidence (Initial Phase) ---")
print(df_pop.head())