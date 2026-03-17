import subprocess, pandas as pd, sys, os, glob, time

print("--- 🏛️ DELFINO ---")

CONFIG = {
    # "total_patients": 7143,    
    "total_patients": 80,    
    "num_workers": 2,         # 2 good on my Laptop
    "seed_offset": 42,
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
            sys.executable, "generate_trajectories.py",
            "--start_id", str(sid), 
            "--end_id", str(eid),
            "--mode", CONFIG["mode"],
            "--seed_offset", str(CONFIG["seed_offset"])
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
    print("\n" * 2 + "📦 Merging parallel chunks into final master files...")
    
    def merge(status, file_type):
        # Matches temp_manual_control_0_120_incidence.csv etc.
        pattern = f"temp_{CONFIG['mode']}_{status}_*_{file_type}.csv"
        files = glob.glob(pattern)
        
        if files:
            # We sort by 'PatientID' (the new column name) to keep the twin cohorts aligned
            df = pd.concat([pd.read_csv(f) for f in files]).sort_values(by="PatientID")
            output_name = f"final_{status}_{file_type}.csv"
            df.to_csv(output_name, index=False)
            
            # Cleanup the temp chunks to keep your folder clean
            for f in files: 
                os.remove(f)
            print(f" - Created {output_name} from {len(files)} chunks.")

    # Loop through both trial arms and both data types
    for status in ["control", "treated"]:
        merge(status, "incidence")
        merge(status, "trajectories")

    print("-" * 40)
    print(f"🏁 PERFORMANCE SUMMARY:")
    print(f"   Total Duration: {total_duration:.2f} seconds")
    print(f"   Throughput:     {throughput:.2f} patients/sec")
    print("-" * 40)

if __name__ == "__main__":
    run()