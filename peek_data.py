import numpy as np

# Define parameters
FILE_TO_PEEK = 'data/ukb_simulated_data/train_smoking.bin'
BLOCK_SIZE = 48

# Load the new heir's data
data = np.fromfile(FILE_TO_PEEK, dtype=np.uint16)
patients = data.reshape(-1, BLOCK_SIZE)

print(f"--- DELFINO VERIFICATION REPORT ---")
print(f"Total Patients in File: {len(patients)}")

# Check the first 5 patients (The ones we modified)
print("\n[Modified Cohort - Expected Token 4 at Index 2]")
for i in range(5):
    val = patients[i, 2]
    status = "PASS" if val == 4 else "FAIL"
    print(f"Patient {i:04d} | Index 2 Token: {val} | {status}")

# Check 5 patients from the middle (The ones we left alone)
print("\n[Control Cohort - Expected Original Tokens]")
for i in range(1000, 1005):
    val = patients[i, 2]
    print(f"Patient {i:04d} | Index 2 Token: {val}")

print("\nVerification Complete.")