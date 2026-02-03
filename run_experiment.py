import subprocess
import pandas as pd
import sys

# --- EXPERIMENT SETTINGS ---
# Change these once to update both Baseline and Intervention runs
CONFIG = {
    "num_patients": 500,
    "time_horizon": 20,
    "start_age": 45.0,
    "seed_offset": 42,
    "print_trajectories": "false" # Set to "true" if you want to see logs
}

def run_experiment():
    print(f"🔔 Starting Delfino Experiment: N={CONFIG['num_patients']}, T={CONFIG['time_horizon']}")

    # Common arguments list
    common_args = [
        "--num_patients", str(CONFIG['num_patients']),
        "--time_horizon", str(CONFIG['time_horizon']),
        "--start_age", str(CONFIG['start_age']),
        "--seed_offset", str(CONFIG['seed_offset'])
    ]
    if CONFIG['print_trajectories'].lower() == "true":
        common_args.append("--print_trajectories")

    # 1. Run Baseline
    print("\n--- Phase 1: Running Baseline ---")
    subprocess.run([sys.executable, "delfino.py"] + common_args, check=True)

    # 2. Run Intervention (add the --apply_intervention flag)
    print("\n--- Phase 2: Running Intervention ---")
    subprocess.run([sys.executable, "delfino.py"] + common_args + ["--apply_intervention"], check=True)

    # 3. Analyze Results
    print("\n📊 Aggregating Results...")
    base = pd.read_csv("delfino_individual_base.csv")
    glp1 = pd.read_csv("delfino_individual_glp1.csv")

    averted = base['DALYs'].sum() - glp1['DALYs'].sum()
    cost_diff = glp1['Cost'].sum() - base['Cost'].sum()

    print("="*40)
    print(f"Total DALYs Averted: {averted:.2f}")
    print(f"Net Program Cost:   ${cost_diff:,.2f}")
    if averted > 0:
        print(f"ICER (Cost/DALY):   ${cost_diff/averted:,.2f}")
    print("="*40)

if __name__ == "__main__":
    run_experiment()