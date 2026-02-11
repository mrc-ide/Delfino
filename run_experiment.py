import subprocess, pandas as pd, sys, os, glob, time

print("--- 🏛️ DELFINO MASTER v2.6 (Parallel + Error Guard) ---")

CONFIG = {
    "total_patients": 500,    
    "num_workers": 2,         
    "time_horizon": 40,
    "start_age": 40.0,
    "logit_bias": 0.0,
    "pin_identity": "true",
    "remind_bmi": "true",
    "seed_offset": 42
}

def run():
    # 1. CLEANUP
    for f in glob.glob("temp_*.csv"):
        try: os.remove(f)
        except: pass

    chunk = CONFIG["total_patients"] // CONFIG["num_workers"]
    procs = []
    print(f"🚀 Launching {CONFIG['num_workers']*2} parallel workers...")

    # 2. SPAWN
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

    print(f"⏳ Monitoring parallel streams...\n" + "\n" * (CONFIG["num_workers"] * 2))
    for p in procs: p.wait()

    # 3. FAILURE GUARD
    temp_files = glob.glob("temp_*.csv")
    if len(temp_files) < (CONFIG["num_workers"] * 2):
        print(f"\n❌ CRITICAL ERROR: Only {len(temp_files)} output files found.")
        print("One or more workers crashed. Check the error logs above!")
        return

    print("\n" * 2 + "📦 Merging and analyzing...")
    def merge(prefix, out):
        files = glob.glob(f"temp_{prefix}_*.csv")
        pd.concat([pd.read_csv(f) for f in files]).sort_values(by="ID").to_csv(out, index=False)
        for f in files: os.remove(f)

    merge("base", "delfino_individual_base.csv")
    merge("glp1", "delfino_individual_glp1.csv")

    subprocess.run([sys.executable, "compare_results.py"], check=True)
    subprocess.run([sys.executable, "plot_results.py", "--start_age", str(CONFIG["start_age"]), "--horizon", str(CONFIG["time_horizon"])], check=True)

if __name__ == "__main__":
    start = time.time()
    run()
    print(f"⏱️ Total Execution Time: {time.time() - start:.2f}s")