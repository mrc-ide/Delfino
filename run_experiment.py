import subprocess, pandas as pd, sys, os, glob, time

print("--- 🏛️  DELFINO ---")

CONFIG = {
    "total_patients": 7143,    
    # "total_patients": 200,    
    "num_workers": 2,         # 2 good on my Laptop
    "seed_offset": 42,
    # "strategy": "on_diagnosis", # choices:  "always", on_diagnosis
    "strategy": "always", # choices:  "always", on_diagnosis
    "trigger_codes": "E66,E11,E67",
    # "trigger_codes": "E66",
    "mode": "manual"
}

def run():
    for f in glob.glob("temp_*.csv"):
        try: os.remove(f)
        except: pass

    # Calculate slices
    chunk = CONFIG["total_patients"] // CONFIG["num_workers"]
    procs = []
    
    start_time = time.time()
    print(f"🚀 Launching {CONFIG['num_workers']*2} workers parallel...")

    current_pos = 0
    for i in range(CONFIG["num_workers"]):
        sid = i * chunk
        eid = (i + 1) * chunk if i < CONFIG["num_workers"] - 1 else CONFIG["total_patients"]
        
        base_cmd = [
            sys.executable, "delfino.py",
            "--start_id", str(sid), 
            "--end_id", str(eid),
            "--mode", CONFIG["mode"],
            "--seed_offset", str(CONFIG["seed_offset"]), 
            "--strategy", CONFIG["strategy"],
            "--trigger_codes", CONFIG["trigger_codes"]
        ]

        # Launch Control - Position 0, 2, 4...
        procs.append(subprocess.Popen(base_cmd + ["--apply_intervention", "False", "--position", str(current_pos)]))
        current_pos += 1

        # Launch Treated - Position 1, 3, 5...
        procs.append(subprocess.Popen(base_cmd + ["--apply_intervention", "True", "--position", str(current_pos)]))
        current_pos += 1


    print(f"⏳ Monitoring {len(procs)} parallel streams..." + "" * (CONFIG["num_workers"] * 2))
    for p in procs: p.wait()

    end_time = time.time()
    total_duration = end_time - start_time
    # Throughput calculation
    throughput = (CONFIG["total_patients"] * 2) / total_duration

    # Merge 
    print("\n" * 2 + f"📦 Merging chunks for patients {0} to {CONFIG['total_patients']}...")
    
    def merge(status, file_type, total_sid, total_eid):
        # Pattern to find all the temporary chunk files
        pattern = f"temp_{CONFIG['mode']}_{status}_*_{file_type}.csv"
        files = glob.glob(pattern)
        
        if files:
            # Sort by PatientID to maintain Digital Twin alignment
            df = pd.concat([pd.read_csv(f) for f in files]).sort_values(by="PatientID")
            
            # NEW NAMING CONVENTION: {status}_{start}_{end}_{type}.csv
            output_name = f"{status}_{total_sid}_{total_eid}_{file_type}.csv"
            df.to_csv(output_name, index=False)
            
            # Cleanup
            for f in files: 
                os.remove(f)
            print(f" - Created: {output_name}")

    # Execute the merge for both trial arms
    # 1. Determine the specific status tag used by generate_trajectories.py
    if CONFIG["strategy"] == "always":
        treat_status = "treated_always"
    else:
        # Replicates the naming logic in the generator script
        safe_codes = CONFIG["trigger_codes"].replace(",", "-")
        treat_status = f"treated_{CONFIG['strategy']}_{safe_codes}"

    # 2. Update the loop to use the dynamic treat_status
    for status in ["control", treat_status]:
        merge(status, "incidence", 0, CONFIG["total_patients"])
        merge(status, "trajectories", 0, CONFIG["total_patients"])

    # ---  PLOTTING  ---
    print("\n" + "📊 Generating Cumulative Incidence plots...")
    
    subprocess.run([
        sys.executable, "plot_results.py", 
        "--start_id", "0", 
        "--end_id", str(CONFIG["total_patients"]),
        "--strategy", CONFIG["strategy"],
        "--trigger_codes", CONFIG["trigger_codes"]
    ], check=True)

    # ---  COMPARISON / ECONOMICS  ---
    print("\n📊 Calculating Cost-Effectiveness (ICER)...")
    subprocess.run([
        sys.executable, "compare_results.py", 
        "--start_id", "0", 
        "--end_id", str(CONFIG["total_patients"]),
        "--strategy", CONFIG["strategy"],
        "--trigger_codes", CONFIG["trigger_codes"]
    ], check=True)


    print(f"✅ Trial Complete. See plots for range 0-{CONFIG['total_patients']}")

    print("-" * 40)
    print(f"🏁 PERFORMANCE SUMMARY:")
    print(f"   Total Duration: {total_duration:.2f} seconds")
    print(f"   Throughput:     {throughput:.2f} patients/sec")
    print("-" * 40)

if __name__ == "__main__":
    run()