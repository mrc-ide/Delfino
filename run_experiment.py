import subprocess
import pandas as pd
import sys
import os

CONFIG = {
    "num_patients": 500,
    "time_horizon": 20,
    "start_age": 45.0,
    "seed_offset": 42,
    "use_real_data": "true"
}

def run_experiment():
    print(f"🔔 Starting Experiment: N={CONFIG['num_patients']}, T={CONFIG['time_horizon']}")
    
    args = [
        "--num_patients", str(CONFIG['num_patients']),
        "--time_horizon", str(CONFIG['time_horizon']),
        "--start_age", str(CONFIG['start_age']),
        "--use_real_data", CONFIG['use_real_data'],
        "--seed_offset", str(CONFIG['seed_offset'])
    ]

    # Phase 1 & 2
    subprocess.run([sys.executable, "delfino.py"] + args, check=True)
    subprocess.run([sys.executable, "delfino.py"] + args + ["--apply_intervention"], check=True)

    # Phase 3: Detailed Comparison
    print("\n📊 Generating Comparison Summary...")
    base = pd.read_csv("delfino_individual_base.csv")
    glp1 = pd.read_csv("delfino_individual_glp1.csv")

    summary = {
        "Metric": ["Total Cost", "Total DALYs"],
        "Baseline": [base['Cost'].sum(), base['DALYs'].sum()],
        "Intervention": [glp1['Cost'].sum(), glp1['DALYs'].sum()]
    }

    # Compare incidence for all diseases
    for col in base.columns:
        if col.startswith("inc_"):
            # Count how many patients got the disease (Age > 0, excludes -99 Pre-existing)
            b_count = (base[col] > 0).sum()
            g_count = (glp1[col] > 0).sum()
            summary["Metric"].append(f"Cases: {col[4:]}")
            summary["Baseline"].append(b_count)
            summary["Intervention"].append(g_count)

    df_summary = pd.DataFrame(summary)
    df_summary["Delta"] = df_summary["Baseline"] - df_summary["Intervention"]
    
    # Output to File
    df_summary.to_csv("delfino_comparison_summary.csv", index=False)
    print("="*40)
    print(df_summary.to_string(index=False))
    print("="*40)
    print(f"📁 Summary saved to delfino_comparison_summary.csv")

if __name__ == "__main__":
    run_experiment()