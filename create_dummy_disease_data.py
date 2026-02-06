import pandas as pd
import os
import re

labels_path = os.path.join('data', 'ukb_simulated_data', 'labels.csv')

def to_medical_sentence_case(text):
    text = text.strip('() ')
    words = text.lower().split()
    if not words: return ""
    words[0] = words[0].capitalize()
    return " ".join(words)

disease_params = []
with open(labels_path, 'r') as f:
    lines = [line.strip() for line in f.readlines()]

# Regex: Starts with a Letter followed by 2-3 digits (Standard ICD-10)
icd_pattern = re.compile(r'^([A-Z][0-9]{2,3})')

for idx, full_string in enumerate(lines):
    match = icd_pattern.match(full_string)
    if match:
        code_part = match.group(1)
        # Remove the code from the start to get the name
        raw_name = full_string[len(code_part):].strip()
        clean_name = to_medical_sentence_case(raw_name)
        
        # Determine weight/cost placeholders based on code (Example: C for Cancer is higher)
        weight = 0.25 if code_part.startswith('C') else 0.1
        cost = 15000 if code_part.startswith('C') else 1000
        
        disease_params.append({
            "TokenID": idx,
            "Code": code_part,
            "Name": clean_name,
            "Cost": cost,
            "Weight": weight
        })

df = pd.DataFrame(disease_params)
df.to_csv('dummy_disease_params.csv', index=False)
print(f"✅ Generated params for {len(df)} diseases (including C-codes).")