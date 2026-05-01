import os
import numpy as np
import pandas as pd
from utils import get_p2i

def check_raw_data():
    # Setup paths
    data_dir = os.path.join('data', 'ukb_simulated_data')
    train_path = os.path.join(data_dir, 'train.bin')
    labels_path = os.path.join(data_dir, 'labels.csv')

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    # 1. Load Labels
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # 2. Load 3-column uint32 Data
    print(f"Loading {train_path} as 3-column uint32...")
    train = np.fromfile(train_path, dtype=np.uint32).reshape(-1, 3)

    # 3. Create Mapping
    p2i = get_p2i(train)
    print(f"Mapping complete. Total patient records found: {len(p2i)}")

    # 4. Extract and Print first 50 Patients
    results = []
    print("\n" + "="*50)
    print("READING FIRST 50 PATIENT RECORDS")
    print("="*50)

    for idx in range(min(51, len(p2i))):
        start, length = p2i[idx]
        p_data = train[start : start + length]
        patient_id = p_data[0, 0]
        
        history = []
        sex_found = "Not Recorded"
        
        for row in p_data:
            age_yr = row[1] / 365.25
            token_id = row[2]
            name = labels[token_id] if token_id < len(labels) else f"ID:{token_id}"
            
            if name.lower() == 'male': sex_found = 'Male'
            if name.lower() == 'female': sex_found = 'Female'
            
            history.append(f"{age_yr:4.1f}: {name}")
        
        hist_str = "\n".join(history)
        
        # Output to console
        print(f"Index: {idx:2} | Patient ID: {patient_id:4} | Sex: {sex_found}")
        print("-" * 30)
        print(hist_str)
        print("="*50)
        
        results.append({
            "Patient_Index": idx,
            "Patient_ID": int(patient_id),
            "Inferred_Sex": sex_found,
            "Raw_History": hist_str
        })

    # 5. Save to CSV
    output_name = "check_trajectories_results.csv"
    pd.DataFrame(results).to_csv(output_name, index=False)
    print(f"\nVerification complete. Results saved to {output_name}")

if __name__ == "__main__":
    check_raw_data()