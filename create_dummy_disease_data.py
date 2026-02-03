import pandas as pd
import os

# Path to your existing labels
labels_path = os.path.join('data', 'ukb_simulated_data', 'labels.csv')

def to_medical_sentence_case(text):
    """Converts string to Sentence case, preserving lowercase for 'and'."""
    # Strip parentheses and extra whitespace
    text = text.strip('() ')
    # Split into words and lowercase everything
    words = text.lower().split()
    if not words:
        return ""
    # Capitalize only the very first word
    words[0] = words[0].capitalize()
    # Join back together
    return " ".join(words)

disease_params = []

# Read lines directly to avoid CSV delimiter issues
with open(labels_path, 'r') as f:
    lines = [line.strip() for line in f.readlines()]

for idx, full_string in enumerate(lines):
    # Capture actual diseases: look for ICD codes in parentheses
    if '(' in full_string and ')' in full_string:
        parts = full_string.split(' ')
        code_part = parts[0] # e.g., 'E11'
        
        # Extract the description and apply the new formatting
        raw_name = " ".join(parts[1:]) 
        clean_name = to_medical_sentence_case(raw_name)
        
        disease_params.append({
            "TokenID": idx,
            "Code": code_part,
            "Name": clean_name,
            "Cost": 1000.0,   # Placeholder
            "Weight": 0.1     # Placeholder
        })

df = pd.DataFrame(disease_params)
df.to_csv('dummy_disease_params.csv', index=False)
print(f"✅ Generated dummy_disease_params.csv with {len(df)} cleaned disease names.")