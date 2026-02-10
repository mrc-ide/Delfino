import subprocess
import sys

# --- GLOBAL CONFIGURATION ---
CONFIG = {
    "num_patients": 500,
    "time_horizon": 40,
    "start_age": 40.0,
    "logit_bias": 0.0,
    "pin_identity": "true",
    "remind_bmi": "true",
    "seed_offset": 42
}

def run_experiment():
    # Convert CONFIG to a list of CLI arguments
    common_args = [
        "--num_patients", str(CONFIG["num_patients"]),
        "--time_horizon", str(CONFIG["time_horizon"]),
        "--start_age", str(CONFIG["start_age"]),
        "--logit_bias", str(CONFIG["logit_bias"]),
        "--pin_identity", CONFIG["pin_identity"],
        "--remind_bmi", CONFIG["remind_bmi"],
        "--seed_offset", str(CONFIG["seed_offset"])
    ]

    print(f"🔔 Starting Orchestration: N={CONFIG['num_patients']}, T={CONFIG['time_horizon']}")

    # Phase 1: Simulations
    print("\n▶️ Running Baseline...")
    subprocess.run([sys.executable, "delfino.py"] + common_args, check=True)
    
    print("\n▶️ Running Intervention...")
    subprocess.run([sys.executable, "delfino.py"] + common_args + ["--apply_intervention"], check=True)

    # Phase 2: Post-Processing Comparison
    print("\n▶️ Running Comparison Script...")
    subprocess.run([sys.executable, "compare_results.py"], check=True)

    # Phase 3: Visualization
    print("\n▶️ Running Plotting Script...")
    # We pass age/horizon to the plotter so the axis matches the config
    subprocess.run([sys.executable, "plot_results.py", 
                    "--start_age", str(CONFIG["start_age"]), 
                    "--horizon", str(CONFIG["time_horizon"])], check=True)

    print("\n🏆 Full Project Delfino Pipeline Completed Successfully.")

if __name__ == "__main__":
    run_experiment()