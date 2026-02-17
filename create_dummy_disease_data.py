import pandas as pd
import os
import re

labels_path = os.path.join('data', 'ukb_simulated_data', 'labels.csv')

def to_medical_sentence_case(text):
    # Strips Excel-induced commas and parentheses
    text = text.strip('() ,')
    words = text.lower().split()
    if not words: return ""
    words[0] = words[0].capitalize()
    return " ".join(words)

disease_params = []

# Read lines directly to avoid CSV-parsing issues with commas in names
with open(labels_path, 'r') as f:
    lines = [line.strip() for line in f.readlines()]

# Regex: Letter + 2-3 alphanumeric chars (Handles A00, C34, and CXX)
icd_pattern = re.compile(r'^([A-Z][0-9X]{2,3})')

for idx, full_string in enumerate(lines):
    match = icd_pattern.match(full_string)
    if match:
        code_part = match.group(1)
        
        # Name is everything following the code
        raw_name = full_string[len(code_part):].strip()
        clean_name = to_medical_sentence_case(raw_name)
        
        if not clean_name:
            clean_name = f"Unspecified Condition ({code_part})"

        # --- SEPARATE WEIGHT LOGIC ---
        # Note: DW and Utility are treated as independent empirical values
        if code_part.startswith('C'): # Malignant Neoplasms
            dw = 0.28                # Health loss (GBD 2021)
            utility = 0.70           # HRQoL (EQ-5D)
            cost = 15000
        elif code_part.startswith('I2'): # Ischaemic Heart Disease
            dw = 0.15
            utility = 0.82
            cost = 5000
        elif code_part.startswith('E1'): # Diabetes
            dw = 0.11
            utility = 0.85
            cost = 3000
        else: # Default values for other conditions
            dw = 0.05
            utility = 0.95
            cost = 1000
        
        disease_params.append({
            "TokenID": idx,
            "Code": code_part,
            "Name": clean_name,
            "Cost": cost,
            "DW": dw,         # For DALYs (Health loss)
            "Utility": utility # For QALYs (Quality of life)
        })

df = pd.DataFrame(disease_params)

# Sort alphabetically by ICD-10 code for human-readability
df = df.sort_values(by="Code").reset_index(drop=True)

df.to_csv('dummy_disease_params.csv', index=False)
print(f"✅ Generated and sorted {len(df)} disease parameters with independent DW/Utility columns.")