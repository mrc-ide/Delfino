import os
import pickle
import torch
import numpy as np
import pandas as pd
from model import Delphi, DelphiConfig
from tqdm import tqdm

# --- 1. CONFIGURATION & SEEDING ---
SEED_OFFSET = 42          # Ensures reproducible "twins" for Baseline vs Intervention
NUM_PATIENTS = 100        # Dataset has 22661 max; 100 is good for testing
TIME_HORIZON = 20         # Total years to project into the future
START_AGE = 45.0          # Starting point for the continuous age coordinate
APPLY_INTERVENTION = True # Toggle to apply GLP-1 treatment logic
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- DATA & OUTPUT OPTIONS ---
USE_REAL_DATA = True      # Pulls medical histories from train.bin
PRINT_TRAJECTORIES = True # Narrative log to console (limited to NUM_PATIENTS)
SAVE_TRAJECTORY_FILES = True  # Writes full audit trail to .txt file

# --- 2. DYNAMIC LABEL & DATA LOADING ---
data_dir = os.path.join('data', 'ukb_simulated_data')
labels_path = os.path.join(data_dir, 'labels.csv')

# Use raw line reading to handle descriptive commas in ICD labels
with open(labels_path, 'r') as f:
    labels_list = [line.strip() for line in f.readlines()]

def get_id(label_name):
    """Utility to find the integer ID for a medical string"""
    try: return str(labels_list.index(label_name))
    except ValueError: return None

# Resolve key IDs for the simulation logic
T_BMI_HIGH = get_id('BMI_high')
T_BMI_MID  = get_id('BMI_mid')
T_DIABETES = get_id('E11 (non-insulin-dependent diabetes mellitus)')
T_HF       = get_id('I50 (heart failure)')
T_STROKE   = get_id('I63 (cerebral infarction)')
T_DEATH    = get_id('Death')

# Health Econ placeholders (weights and annual costs)
DISABILITY_WEIGHTS = {T_DIABETES: 0.05, T_HF: 0.18, T_STROKE: 0.23, T_DEATH: 1.0}
DISEASE_COSTS = {T_DIABETES: 2500, T_HF: 8000, T_STROKE: 15000}
GLP_YEARLY_COST = 1200

# Memory-map the binary data for fast access to patient histories
if USE_REAL_DATA:
    all_patients_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)
    print(f"✅ Context Loaded: {len(all_patients_data)//48} histories available.")

# --- 3. MODEL INITIALIZATION ---
out_dir = 'out-delfino-baseline'
checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=DEVICE)

# Extract VOCAB_SIZE from weights to prevent GPU out-of-bounds crashes
VOCAB_SIZE = checkpoint['model_args']['vocab_size']
print(f"🔍 Model Vocab Size: {VOCAB_SIZE}")

model = Delphi(DelphiConfig(**checkpoint['model_args']))
model.load_state_dict(checkpoint['model'])
model.to(DEVICE).eval()

# --- 4. SIMULATOR FUNCTIONS ---
def get_safe_label(token_id):
    """Safely map token ID to label, handling out-of-bounds indices"""
    if token_id < len(labels_list):
        return labels_list[token_id]
    return f"UnknownToken_{token_id}"

def simulate_patient(patient_id, apply_glp1):
    # Unique but reproducible seed for this patient
    np.random.seed(patient_id * SEED_OFFSET) 
    torch.manual_seed(patient_id * SEED_OFFSET)
    
    current_age = START_AGE
    
    if USE_REAL_DATA:
        # Slice the 48-token block for this specific patient
        start_idx = patient_id * 48
        raw_tokens = all_patients_data[start_idx : start_idx + 48].tolist()
        
        # Hardware-safety: Clip tokens to model's vocab size
        sanitized = [t for t in raw_tokens if t < VOCAB_SIZE]
        context_tokens = [t for t in sanitized if get_safe_label(t) not in ["Padding", "No event"]]
        history_log = [f"START: Age {int(current_age)}, {', '.join([get_safe_label(t) for t in context_tokens])}"]
    else:
        # Fallback to Template (Male + High BMI)
        context_tokens = [int(get_id('Male')), int(T_BMI_HIGH)]
        history_log = [f"START: Age {int(current_age)}, Male, BMI High"]

    total_costs, total_dalys, events = 0, 0, []

    for year in range(TIME_HORIZON):
        # Truncate to the Transformer's context window limit
        context_tokens = context_tokens[-48:] 

        # INTERVENTION: Scan for High BMI and apply probability-based treatment
        eligible = False
        for i, token in enumerate(context_tokens):
            if str(token) == T_BMI_HIGH:
                eligible = True
                if apply_glp1 and np.random.rand() < 0.8: # 80% Success Rate
                    context_tokens[i] = int(T_BMI_MID)
                    history_log.append(f"Age {int(current_age)}: GLP-1 (BMI High -> Mid)")
                break 
        
        if apply_glp1 and eligible:
            total_costs += GLP_YEARLY_COST 

        # Blackwell GPU Inference (Context Sequence + Continuous Age)
        x = torch.tensor(context_tokens, dtype=torch.long, device=DEVICE)[None, ...]
        age_t = torch.tensor([current_age], dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            out = model(x, age=age_t)
            logits = out[0] if isinstance(out, (list, tuple)) else out
            probs = torch.softmax(logits[:, -1, :] / 1.0, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        
        # Update sequence with prediction
        context_tokens.append(next_token)
        label = get_safe_label(next_token)
        if label not in ["No event", "Padding"]:
            history_log.append(f"Age {int(current_age)}: {label}")

        # Accrue costs and DALYs if a tracked disease is predicted
        t_str = str(next_token)
        if t_str in DISABILITY_WEIGHTS:
            total_dalys += DISABILITY_WEIGHTS[t_str]
            total_costs += DISEASE_COSTS.get(t_str, 0)
            events.append(label)
        
        if t_str == T_DEATH:
            total_dalys += 15.0 # Fixed YLL placeholder for death
            break
        current_age += 1.0

    narrative = " -> ".join(history_log)
    if PRINT_TRAJECTORIES and patient_id < NUM_PATIENTS:
        print(f"\n--- Patient {patient_id} ---\n{narrative}")

    return {"ID": patient_id, "Cost": total_costs, "DALYs": total_dalys, "Events": "|".join(events), "Full_History": narrative}

# --- 5. EXECUTION & SAVING ---
print(f"🚀 Running {'Intervention' if APPLY_INTERVENTION else 'Baseline'} on Blackwell GPU...")
results = [simulate_patient(i, APPLY_INTERVENTION) for i in tqdm(range(NUM_PATIENTS))]

df = pd.DataFrame(results)
suffix = 'glp1' if APPLY_INTERVENTION else 'base'

# Export lightweight results for analysis
df[['ID', 'Cost', 'DALYs', 'Events']].to_csv(f"delfino_results_{suffix}.csv", index=False)

# Export full text audit trail
if SAVE_TRAJECTORY_FILES:
    with open(f"delfino_trajectories_{suffix}.txt", "w") as f:
        for idx, row in df.iterrows():
            f.write(f"Patient {row['ID']}:\n{row['Full_History']}\n{'-'*50}\n")
    print(f"📁 Narratives saved to delfino_trajectories_{suffix}.txt")

print("\n--- Summary Statistics ---")
print(df[['Cost', 'DALYs']].describe().loc[['mean', 'max']])