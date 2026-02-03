import subprocess
import pandas as pd
import sys
import os

# --- EXPERIMENT SETTINGS ---
CONFIG = {
    "num_patients": 1000,    # Increased to 1000 for better statistical power
    "time_horizon": 30,      # Increased to 30 years to allow diseases to develop
    "start_age": 45.0,
    "seed_offset": 42,
    "use_real_data": "true"
}

def run_experiment():
    print(f"🔔 Starting Delfino Experiment: N={CONFIG['num_patients']}, T={CONFIG['time_horizon']}")
    
    args = ["--num_patients", str(CONFIG['num_patients']), "--time_horizon", str(CONFIG['time_horizon']),
            "--start_age", str(CONFIG['start_age']), "--use_real_data", CONFIG['use_real_data'],
            "--seed_offset", str(CONFIG['seed_offset'])]

    # Phase 1 & 2
    subprocess.run([sys.executable, "delfino.py"] + args, check=True)
    subprocess.run([sys.executable, "delfino.py"] + args + ["--apply_intervention"], check=True)

    # Phase 3: Detailed Comparison
    print("\n📊 Generating Comparison Summary...")
    base = pd.read_csv("delfino_individual_base.csv")
    glp1 = pd.read_csv("delfino_individual_glp1.csv")

    summary_data = []
    
    # 1. Economic Totals
    summary_data.append(["Total Cost", base['Cost'].sum(), glp1['Cost'].sum()])
    summary_data.append(["Total DALYs", base['DALYs'].sum(), glp1['DALYs'].sum()])

    # 2. Disease Incidence (New Cases)
    for col in base.columns:
        if col.startswith("inc_"):
            # Patients who didn't have it at start (-1.0)
            at_risk = (base[col] != -99.0).sum()
            # New cases (recorded an age > 0)
            b_new = (base[col] > 0).sum()
            g_new = (glp1[col] > 0).sum()
            
            summary_data.append([f"At-Risk: {col[4:]}", at_risk, at_risk])
            summary_data.append([f"New Cases: {col[4:]}", b_new, g_new])

    df_summary = pd.DataFrame(summary_data, columns=["Metric", "Baseline", "Intervention"])
    df_summary["Delta"] = df_summary["Baseline"] - df_summary["Intervention"]
    
    # Save to File
    df_summary.to_csv("delfino_comparison_summary.csv", index=False)
    print("="*60)
    print(df_summary.to_string(index=False))
    print("="*60)
    print(f"📁 Comparison saved to delfino_comparison_summary.csv")

if __name__ == "__main__":
    run_experiment()