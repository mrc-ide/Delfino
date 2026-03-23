import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# Official Delphi imports
from model import Delphi, DelphiConfig
from utils import get_p2i, get_batch

# --- SETTINGS ---
# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Delphi Trajectory Generator")
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--end_id', type=int, default=120) # max of 7143
parser.add_argument('--max_new_tokens', type=int, default=100)
# MODE Options: 
# 'manual' (Default): Uses the competing risks race loop directly on x and a.
# 'automatic': Uses the model.generate() wrapper on cloned x and a. # note this currently has no tracking of disease timings, as can't easily input efficacies/logit biases
parser.add_argument('--mode', type=str, default='manual', choices=['manual', 'automatic'])
parser.add_argument('--apply_intervention', type=str, default='True', choices=['True', 'False'])
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--seed_offset', type=int, default=42)
parser.add_argument('--position', type=int, default=0, help="Terminal line for tqdm progress bar")
parser.add_argument('--strategy', type=str, default='always', choices=['always', 'on_diagnosis'])
parser.add_argument('--trigger_codes', type=str, default='E66', help="ICD code that triggers treatment")

args = parser.parse_args()

# extract CL arguments and put them here so you don't have to change rest of your code.
START_ID = args.start_id
END_ID = args.end_id # max of 7143
MAX_NEW_TOKENS = args.max_new_tokens
MODE = args.mode
APPLY_INTERVENTION = args.apply_intervention == 'True'
# APPLY_INTERVENTION = False 
DEVICE = args.device
SEED_OFFSET = args.seed_offset
POSITION = args.position # Map it to a global like you did for others
STRATEGY = args.strategy
TRIGGER_CODES = args.trigger_codes
# standard_life_expectancy
STANDARD_LIFE_EXPECTANCY = 86.0  ## (dummy, not one-size-fits-all)


DAYS_PER_YEAR = 365.25

# Constants derived from args or fixed

# --- INTERVENTION SETTINGS ---
# APPLY_INTERVENTION = False  
# Map ICD-10 Code -> Hazard Ratio (HR)
# e.g., {"I10": 0.5} means 50% reduction in Hypertension incidence
affected_diseases = {
    "I10": 0.001,  # Test "cure" for Hypertension
    "E11": 0.06,   # Diabetes: 94% reduction (SURMOUNT-3)
    "I50": 0.50,   # Heart Failure: 50% reduction (SUMMIT / STEP-HFpEF)
    "I21": 0.78,   # MACE/MI: 22% reduction (SELECT / SUSTAIN-6)
    "I63": 0.78,   # Stroke: 22% reduction (SELECT / SUSTAIN-6)
    "N18": 0.76,   # Chronic Kidney Disease: 24% reduction (FLOW)
}
# Special handling for Mortality: 18% reduction (LEADER/SELECT/FLOW)
DEATH_HR = 0.82

DATA_DIR = os.path.join('data', 'ukb_simulated_data')
TRAIN_PATH = os.path.join(DATA_DIR, 'train.bin')
LABELS_PATH = os.path.join(DATA_DIR, 'labels.csv')
CKPT_PATH = 'out-delfino-baseline/ckpt.pt'
T_DEATH_ID = 1269

def generate_trajectories():

    ### === LOAD INPUTS
    # load Token labels
    with open(LABELS_PATH, 'r') as f:
        labels_list = [line.strip() for line in f.readlines()]

    # HEOR: Load (dummy) disability weights for DALYs, utilities for QALYs, and costs (currently just dummy costs)
    econ_df = pd.read_csv('disease_params_ihme.csv').set_index('TokenID')
    # Map for O(1) lookup: {TokenID: {'Utility': 0.95, 'Cost': 1000, 'DW': 0.05}}
    ECON_LOOKUP = econ_df[['Utility', 'Cost', 'DW']].to_dict('index')
    # Specific intervention cost for the trial
    DRUG_ANNUAL_COST = 1200.0 # GLP-1 therapy cost per year (dummy)

    # Identify all ICD-10 codes (Letter followed by numbers)
    # Mapping: {TokenID: "Code"}
    TRACKED_CODES = {}
    for i, label in enumerate(labels_list):
        # Match codes like I50, E11, but skip 'Padding' or 'No event'
        if len(label) >= 3 and label[0].isalpha() and label[1].isdigit():
            # Extract just the code part (e.g., "I50" from "I50 (heart failure)")
            code = label.split(' ')[0]
            TRACKED_CODES[i] = code

    # Reverse map to find indices for the affected codes
    code_to_id = {v: k for k, v in TRACKED_CODES.items()}

    # Distinct list of unique codes for CSV columns
    unique_codes = sorted(list(set(TRACKED_CODES.values())))

    # Vocabulary size is len(labels_list)
    logit_bias_vector = torch.zeros(len(labels_list), device=DEVICE)

    if APPLY_INTERVENTION:
        # print(f"Applying Intervention on {len(affected_diseases)} disease(s):")
        # Build a SET of IDs to support multiple trigger codes (e.g., "E66,E11")
        # We split by comma and strip whitespace
        target_trigger_codes = [c.strip() for c in TRIGGER_CODES.split(",")]
        trigger_id_set = {code_to_id[c] for c in target_trigger_codes if c in code_to_id}
        print(f"Monitoring for Trigger Codes: {target_trigger_codes} (IDs: {trigger_id_set})")
        
        for code, hr in affected_diseases.items():
            if code in code_to_id:
                tid = code_to_id[code]
                bias = np.log(hr)
                logit_bias_vector[tid] = bias
                # print(f" - {code} (ID: {tid}): HR={hr} (Logit Bias: {bias:.4f})")
            # else:
                # print(f" - Warning: {code} not found in labels.")
        # Explicitly apply the mortality benefit to the Death Token
        death_bias = np.log(DEATH_HR)
        logit_bias_vector[T_DEATH_ID] = death_bias
        # print(f" - Death (ID: {T_DEATH_ID}): HR={DEATH_HR} (Logit Bias: {death_bias:.4f})")

    # create containers for results
    trajectories = {}
    # Container for quantitative results (as opposed to string trajectories)
    all_metrics = []

    # load model checkpoint (weights)
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model = Delphi(DelphiConfig(**checkpoint['model_args'])).to(DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Load training data as uint32 triplets [PID, Age, Token] in 3-columns
    train_data = np.fromfile(TRAIN_PATH, dtype=np.uint32).reshape(-1, 3)
    
    # Get the patient to index mapping.
    p2i = get_p2i(train_data)

    # print(f"Running generation in {MODE} mode...")

    ### === Begin person loop
    for pid in tqdm(range(START_ID, END_ID), position=POSITION, leave=True, desc=f"Chunk {POSITION}"):

        # 1. Check if patient ALREADY meets criteria in their history
        # (Using the p2i lookup we already have)
        # trigger_id = code_to_id.get(TRIGGER_CODES)
        
        total_costs = 0.0
        total_qalys = 0.0
        total_ylds = 0.0  # Disability burden (YLD)
        total_ylls = 0.0  # Mortality burden (YLL)

        # SEEDING FOR each digital twin
        # Seed both Numpy and Torch for reproducibility
        # Adding a large constant (SEED_OFFSET) prevents potential overlap with other seeds
        torch.manual_seed(pid + SEED_OFFSET)
        np.random.seed(pid + SEED_OFFSET)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(pid + SEED_OFFSET)
        
        # skip if specific person id is out of range of data.
        if pid >= len(p2i): continue
            
        # Get person's input context using delphi batching logic (includes +1 shift)
        x, a, _, _ = get_batch(ix=[pid], data=train_data, p2i=p2i, select='left', 
                              block_size=128, device=DEVICE, padding='random', no_event_token_rate=5)
        
        # Check for the drug trigger
        history_tokens = set(x[0].cpu().numpy().tolist())
        current_chronic_ids = [int(tid) for tid in x[0].cpu().numpy() if int(tid) in ECON_LOOKUP]
        
        # Drug is active if strategy is 'always' OR any trigger ID is in history
        if APPLY_INTERVENTION:
            drug_active = (STRATEGY == 'always') or not trigger_id_set.isdisjoint(history_tokens) # i.e. any overlap between trigger_id_set and history_tokens (forgive the double negative as it's more computationally efficient)

        # Initialize record with -1.0 (Absence)
        # SimulationStartAge is the age of the very last token in history
        start_age_y = a[0, -1].item() / DAYS_PER_YEAR
        inc_record = {
            "PatientID": pid, 
            "SimulationStartAge": start_age_y,
            **{code: -1.0 for code in unique_codes}
        }

        # Begin caculating trajectories
        if MODE == 'manual':
            # Direct pass: no cloning, uses the original tensors from get_batch
            curr_x, curr_a = x, a 
            manual_tokens = []
            manual_ages = []

            for _ in range(MAX_NEW_TOKENS):
                with torch.no_grad():
                    # Forward pass to get Hazard Rates (logits)
                    out = model(curr_x, age=curr_a)
                    logits = out[0][:, -1, :] 
                    
                    # --- VECTORIZED INTERVENTION GATE ---
                    if APPLY_INTERVENTION:
                        # Adding the vector (mostly zeros) to the logits
                        if drug_active:
                            logits += logit_bias_vector
                    # ------------------------------------

                    # Competing Risks Race: Sample wait times from exponential distribution
                    # Inverse CDF method: T = -1/lambda * ln(U)
                    t_wait = torch.clamp(-torch.exp(-logits) * torch.rand(logits.shape, device=DEVICE).log(), min=0)
                    t_next = t_wait.min(1) # [0] is time, [1] is index
                    
                    next_id = t_next[1][:, None]
                    next_age = curr_a[..., [-1]] + t_next[0][:, None]

                    # Years passed in this step
                    dt = t_wait.min(1)[0].item()

                    # Integration: QALYs and Maintenance Costs
                    current_u = 1.0
                    current_dw_complement = 1.0 # Complement for multiplicative DW
                    for tid in current_chronic_ids:
                        current_u *= ECON_LOOKUP[tid]['Utility']
                        current_dw_complement *= (1.0 - ECON_LOOKUP[tid]['DW'])
                    total_qalys += (current_u * dt)
                    current_dw_combined = 1.0 - current_dw_complement
                    total_ylds += (current_dw_combined * dt)

                    # Accumulate Maintenance Costs
                    # Includes annual cost of diseases + drug cost (if active)
                    maint_tick = 0.0
                    if APPLY_INTERVENTION and drug_active:
                        maint_tick += DRUG_ANNUAL_COST
                    
                    for tid in current_chronic_ids:
                        maint_tick += ECON_LOOKUP[tid]['Cost']
                    
                    total_costs += (maint_tick * dt)
                    
                    # Check for NEW Trigger (if not already active)
                    token_id = next_id.item()
                    if APPLY_INTERVENTION:
                        # if not drug_active and new_tid == trigger_id:
                        if not drug_active and token_id in trigger_id_set:
                            drug_active = True
                            # Optional: Add a 'prescription cost' here
                    
                    # TRACKING: If token is a disease and not yet recorded, save the age
                    if token_id in TRACKED_CODES:
                         code = TRACKED_CODES[token_id]
                         if inc_record[code] == -1.0:
                             inc_record[code] = next_age.item() / DAYS_PER_YEAR

                    # Add new diagnosis to list for future maintenance/utility impact
                    if token_id in ECON_LOOKUP and token_id not in current_chronic_ids:
                        current_chronic_ids.append(token_id)

                manual_tokens.append(next_id.item())
                manual_ages.append(next_age.item())

                # Update context for the next step in the loop
                curr_x = torch.cat([curr_x, next_id], dim=1)
                curr_a = torch.cat([curr_a, next_age], dim=1)

                # Stop if the "winner" is Death
                if next_id.item() == T_DEATH_ID:
                    age_at_death = next_age.item() / DAYS_PER_YEAR
                    total_ylls = max(0, STANDARD_LIFE_EXPECTANCY - age_at_death)
                    break
            
            # For manual, the full trajectory is now in the updated curr_x/curr_a
            gen_tokens = curr_x[0].cpu().numpy()
            gen_ages = curr_a[0].cpu().numpy()
            input_len = x.shape[1]

        elif MODE == 'automatic':
            # Wrapper pass: Clones tensors so original x and a tensors are preserved
            with torch.no_grad():
                y, b, _ = model.generate(x.clone(), a.clone(), 
                                         max_new_tokens=MAX_NEW_TOKENS, 
                                         termination_tokens=[T_DEATH_ID])
            gen_tokens = y[0].cpu().numpy()
            gen_ages = b[0].cpu().numpy()
            input_len = x.shape[1]

        # Join patient history (input trajectory) to predicted diseases (generated trajectory)
        lines = ["Input trajectory:"]
        for i in range(len(gen_tokens)):
            # Divider between History and Generated Future
            if i == input_len:
                lines.append("=====================")
                lines.append(f"{MODE.capitalize()} Generated trajectory:")
            
            tid = int(gen_tokens[i])
            age_y = gen_ages[i] / DAYS_PER_YEAR

            # SKIP PADDING: Fixes the "weirdness" at the start of trajectories
            if tid == 0: 
                continue 
            
            # Map ID 1:1 to labels.csv index
            event_name = labels_list[tid] if tid < len(labels_list) else f"Unknown({tid})"
            lines.append(f"{age_y:2.1f}: {event_name}")
            
            # Stop display if terminal token is reached
            if i >= input_len and tid == T_DEATH_ID:
                break
                
        # add this person's lines to trajectories
        trajectories[str(pid)] = "\n".join(lines)

        # Append the record to the master list after each patient is done
        inc_record.update({
            "Total_Costs": total_costs,
            "Total_QALYs": total_qalys,
            "Total_YLDs": total_ylds,
            "Total_YLLs": total_ylls,
            "Total_DALYs": total_ylds + total_ylls # Total burden
        })
        all_metrics.append(inc_record)

    ### === SAVE OUTPUTS
    # status = "treated" if APPLY_INTERVENTION else "control"
    if not APPLY_INTERVENTION:
        status = "control"
    else:
        if STRATEGY == 'always':
            status = "treated_always"
        else:
            # e.g., "treated_on_diagnosis_E66-E11"
            safe_codes = TRIGGER_CODES.replace(",", "-")
            status = f"treated_{STRATEGY}_{safe_codes}"

    # Results will save as:
    # - control_0_200_incidence.csv
    # - treated_always_0_200_incidence.csv
    # - treated_on_diagnosis_E66-E11_0_200_incidence.csv

    # Save string trajectories
    trajectories_output_filename = f"temp_{MODE}_{status}_{START_ID}_{END_ID}_trajectories.csv"
    df_results = pd.DataFrame(list(trajectories.items()), columns=["PatientID", "Trajectory"])
    df_results.to_csv(trajectories_output_filename, index=False)

    # Save the Incidence CSV
    incidence_filename = f"temp_{MODE}_{status}_{START_ID}_{END_ID}_incidence.csv"
    df_incidence = pd.DataFrame(all_metrics)
    # Reorder columns to put PatientID and StartAge first
    # cols = ["PatientID", "SimulationStartAge"] + unique_codes
    # cols = ["PatientID", "SimulationStartAge", "Total_Costs", "Total_QALYs"] + unique_codes
    metrics_cols = ["Total_Costs", "Total_QALYs", "Total_YLDs", "Total_YLLs", "Total_DALYs"]
    cols = ["PatientID", "SimulationStartAge"] + metrics_cols + unique_codes
    df_incidence = df_incidence[cols]
    df_incidence.to_csv(incidence_filename, index=False)

    # print(f"\nDone. Results saved with status '{status}' to:")
    # print(f" - {trajectories_output_filename}")
    # print(f" - {incidence_filename}")

if __name__ == "__main__":
    generate_trajectories()