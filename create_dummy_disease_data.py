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

        # Severity Logic
        if code_part.startswith('C'):
            weight = 0.25 
            cost = 15000
        elif code_part.startswith('E1'):
            weight = 0.15
            cost = 3000
        else:
            weight = 0.1
            cost = 1000
        
        disease_params.append({
            "TokenID": idx,
            "Code": code_part,
            "Name": clean_name,
            "Cost": cost,
            "Weight": weight
        })

df = pd.DataFrame(disease_params)

# --- 🗃️ ALPHABETICAL SORTING ---
# This ensures A00 is at the top and CXX/D48 are further down
df = df.sort_values(by="Code").reset_index(drop=True)

df.to_csv('dummy_disease_params.csv', index=False)
print(f"✅ Generated and alphabetically sorted {len(df)} disease parameters.")