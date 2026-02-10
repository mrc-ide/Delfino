import subprocess, pandas as pd, sys, os

CONFIG = {
    "num_patients": 1000,
    "time_horizon": 40,
    "start_age": 40.0,
    "logit_bias": 0.0,
    "pin_identity": "true",
    "remind_bmi": "true"
}

def run():
    args = ["--num_patients", str(CONFIG['num_patients']), "--time_horizon", str(CONFIG['time_horizon']),
            "--start_age", str(CONFIG['start_age']), "--logit_bias", str(CONFIG['logit_bias']),
            "--pin_identity", CONFIG['pin_identity'], "--remind_bmi", CONFIG['remind_bmi']]

    # Run Simulation
    subprocess.run([sys.executable, "delfino.py"] + args, check=True)
    subprocess.run([sys.executable, "delfino.py"] + args + ["--apply_intervention"], check=True)

    # Post-Process
    base = pd.read_csv("delfino_individual_base.csv")
    glp = pd.read_csv("delfino_individual_glp1.csv")

    metrics = ["Cost", "YLD", "YLL", "DALYs", "QALYs_Add", "QALYs_Mult"]
    summary = []
    for m in metrics:
        summary.append([m, base[m].mean(), glp[m].mean()])

    df_sum = pd.DataFrame(summary, columns=["Metric", "Base", "GLP1"])
    df_sum["Delta"] = df_sum["GLP1"] - df_sum["Base"]
    
    # Simple ICER
    q_gain = df_sum.loc[df_sum['Metric'] == 'QALYs_Mult', 'Delta'].values[0]
    c_inc = df_sum.loc[df_sum['Metric'] == 'Cost', 'Delta'].values[0]
    
    df_sum.to_csv("delfino_comparison_summary.csv", index=False)
    print("\n--- Summary Results ---")
    print(df_sum.to_string(index=False))
    if q_gain > 0: print(f"\n💰 ICER: £{c_inc / q_gain:,.2f} per QALY gained")

if __name__ == "__main__":
    run()