import subprocess
import pandas as pd
import sys
import os

# --- EXPERIMENT SETTINGS ---
CONFIG = {
    "num_patients": 500,
    "time_horizon": 20,
    "start_age": 45.0,
    "seed_offset": 42,
    "use_real_data": "true",
    "print_trajectories": "false",
    "num_to_print": 10,
    "save_trajectories": "true"
}

def run_experiment():
    print(f"🔔 Starting Delfino Experiment: N={CONFIG['num_patients']}, T={CONFIG['time_horizon']}")

    # Build Arguments List
    common_args = [
        "--num_patients", str(CONFIG['num_patients']),
        "--time_horizon", str(CONFIG['time_horizon']),
        "--start_age", str(CONFIG['start_age']),
        "--seed_offset", str(CONFIG['seed_offset']),
        "--use_real_data", CONFIG['use_real_data'],
        "--num_to_print", str(CONFIG['num_to_print']),
        "--save_trajectories", CONFIG['save_trajectories']
    ]
    
    # Flags (no value needed)
    if CONFIG['print_trajectories'].lower() == "true":
        common_args.append("--print_trajectories")

    try:
        # 1. Run Baseline
        print("\n--- Phase 1: Running Baseline ---")
        subprocess.run([sys.executable, "delfino.py"] + common_args, check=True)

        # 2. Run Intervention
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
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiment Failed: The simulation engine returned an error.")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")

if __name__ == "__main__":
    run_experiment()