import pandas as pd
import os
import re

labels_path = os.path.join('data', 'ukb_simulated_data', 'labels.csv')

def to_medical_sentence_case(text):
    text = text.strip('() ,')
    words = text.lower().split()
    if not words: return ""
    words[0] = words[0].capitalize()
    return " ".join(words)

disease_params = []
with open(labels_path, 'r') as f:
    lines = [line.strip() for line in f.readlines()]

icd_pattern = re.compile(r'^([A-Z][0-9X]{2,3})')

for idx, full_string in enumerate(lines):
    match = icd_pattern.match(full_string)
    if match:
        code_part = match.group(1)
        raw_name = full_string[len(code_part):].strip()
        clean_name = to_medical_sentence_case(raw_name)
        if not clean_name: clean_name = f"Unspecified Condition ({code_part})"

        # -- SEPARATE WEIGHT LOGIC --
        if code_part.startswith('C'): # Cancer
            dw = 0.28                 # Disability Weight (GBD Style)
            utility = 0.68            # Utility (EQ-5D Style - note it's not exactly 1-0.28)
            cost = 15000
        elif code_part.startswith('E1'): # Diabetes
            dw = 0.12
            utility = 0.84
            cost = 3000
        else: # Default
            dw = 0.08
            utility = 0.90
            cost = 1000
        
        disease_params.append({
            "TokenID": idx,
            "Code": code_part,
            "Name": clean_name,
            "Cost": cost,
            "DW": dw,         # For DALYs
            "Utility": utility # For QALYs
        })

df = pd.DataFrame(disease_params).sort_values(by="Code").reset_index(drop=True)
df.to_csv('dummy_disease_params.csv', index=False)
print(f"✅ Generated decoupled weights for {len(df)} diseases.")