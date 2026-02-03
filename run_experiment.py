import subprocess, pandas as pd, sys, os

CONFIG = {
    "num_patients": 1000,
    "time_horizon": 30,
    "start_age": 45.0,
    "seed_offset": 42,
    "logit_bias": -3.0 
}

def run_experiment():
    print(f"🔔 Starting Experiment: N={CONFIG['num_patients']}, T={CONFIG['time_horizon']}")
    args = ["--num_patients", str(CONFIG['num_patients']), "--time_horizon", str(CONFIG['time_horizon']),
            "--start_age", str(CONFIG['start_age']), "--seed_offset", str(CONFIG['seed_offset']),
            "--logit_bias", str(CONFIG['logit_bias'])]

    # Phase 1 & 2: Sequential Simulation Runs
    subprocess.run([sys.executable, "delfino.py"] + args, check=True)
    subprocess.run([sys.executable, "delfino.py"] + args + ["--apply_intervention"], check=True)

    # Phase 3: Robust Comparison
    print("\n📊 Aggregating Results...")
    base = pd.read_csv("delfino_individual_base.csv")
    glp = pd.read_csv("delfino_individual_glp1.csv")

    summary = [["Total Cost", base['Cost'].sum(), glp['Cost'].sum()],
               ["Total DALYs", base['DALYs'].sum(), glp['DALYs'].sum()]]

    # Union of all incidence columns found in either run
    all_inc_cols = sorted(list(set([c for c in base.columns if c.startswith('inc_')]) | 
                               set([c for c in glp.columns if c.startswith('inc_')])))

    for col in all_inc_cols:
        b_cnt = (base[col] > 0).sum() if col in base.columns else 0
        g_cnt = (glp[col] > 0).sum() if col in glp.columns else 0
        summary.append([f"Cases: {col[4:]}", b_cnt, g_cnt])

    df_sum = pd.DataFrame(summary, columns=["Metric", "Baseline", "Intervention"])
    df_sum["Delta"] = df_sum["Baseline"] - df_sum["Intervention"]
    df_sum.to_csv("delfino_comparison_summary.csv", index=False)
    print("\n", df_sum.to_string(index=False))

if __name__ == "__main__":
    run_experiment()