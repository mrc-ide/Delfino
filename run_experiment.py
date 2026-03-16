import subprocess, pandas as pd, sys, os, glob, time

print("--- 🏛️ DELFINO MASTER v2.7 ---")

CONFIG = {
    "total_patients": 22661,    
    "num_workers": 2,         # 2 good on my Laptop
    "time_horizon": 20,
    "start_age": 40.0,
    "logit_bias": 0,
    "pin_identity": "true",
    "remind_bmi": "true",
    "seed_offset": 42
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
            "--start_id", str(sid), "--end_id", str(eid),
            "--time_horizon", str(CONFIG["time_horizon"]),
            "--start_age", str(CONFIG["start_age"]),
            "--logit_bias", str(CONFIG["logit_bias"]),
            "--pin_identity", CONFIG["pin_identity"],
            "--remind_bmi", CONFIG["remind_bmi"],
            "--position", str(current_pos)
        ]

        procs.append(subprocess.Popen(base_cmd))
        current_pos += 1
        procs.append(subprocess.Popen(base_cmd + ["--apply_intervention", "--position", str(current_pos)]))
        current_pos += 1

    print(f"⏳ Monitoring {len(procs)} parallel streams..." + "" * (CONFIG["num_workers"] * 2))
    for p in procs: p.wait()

    end_time = time.time()
    total_duration = end_time - start_time
    # Throughput calculation
    throughput = (CONFIG["total_patients"] * 2) / total_duration

    print("\n" * 2 + "📦 Merging and analyzing...")
    
    def merge(prefix, out):
        files = glob.glob(f"temp_{prefix}_*.csv")
        if files:
            pd.concat([pd.read_csv(f) for f in files]).sort_values(by="ID").to_csv(out, index=False)
            for f in files: os.remove(f)

    merge("base", "delfino_individual_base.csv")
    merge("glp1", "delfino_individual_glp1.csv")

    # Call modular Post-Processors
    subprocess.run([sys.executable, "compare_results.py"], check=True)
    subprocess.run([sys.executable, "plot_results.py", "--start_age", str(CONFIG["start_age"]), "--horizon", str(CONFIG["time_horizon"])], check=True)

    print("-" * 40)
    print(f"🏁 PERFORMANCE SUMMARY:")
    print(f"   Total Duration: {total_duration:.2f} seconds")
    print(f"   Throughput:     {throughput:.2f} patients/sec")
    print("-" * 40)

if __name__ == "__main__":
    run()