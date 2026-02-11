import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_age', type=float, default=40.0)
    parser.add_argument('--horizon', type=int, default=40)
    args = parser.parse_args()

    # 1. Load Data
    if not os.path.exists("delfino_individual_base.csv"):
        print("❌ Error: Result files not found. Run the simulation first.")
        return
        
    base = pd.read_csv("delfino_individual_base.csv")
    glp = pd.read_csv("delfino_individual_glp1.csv")
    params_df = pd.read_csv("dummy_disease_params.csv")
    name_map = dict(zip(params_df['Code'], params_df['Name']))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # --- 1. Cumulative Incidence Curves ---
    inc_cols = [c for c in base.columns if c.startswith("inc_") and c != "inc_death"]
    top_5 = sorted(inc_cols, key=lambda x: (base[x] > 0).sum(), reverse=True)[:5]
    
    ages = np.arange(args.start_age, args.start_age + args.horizon + 1)

    for i, d in enumerate(top_5):
        code = d[4:]
        name = name_map.get(code, code)
        # Calculate cumulative cases at each age
        b_val = [(base[d] <= age).where(base[d] > 0).sum() for age in ages]
        g_val = [(glp[d] <= age).where(glp[d] > 0).sum() if d in glp.columns else 0 for age in ages]
        
        ax1.plot(ages, b_val, '--', color=f"C{i}", alpha=0.5, label=f"{name} (Base)")
        ax1.plot(ages, g_val, '-', color=f"C{i}", linewidth=2, label=f"{name} (GLP-1)")

    ax1.set_title(f"Cumulative Disease Incidence (N={len(base)}, T={args.horizon})", fontsize=14)
    ax1.set_xlabel("Patient Age")
    ax1.set_ylabel("Total Number of Cases")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- 2. QALY Bar Chart ---
    x_pos = np.array([0, 1])
    width = 0.35
    
    base_means = [base['QALYs_Add'].mean(), base['QALYs_Mult'].mean()]
    glp_means = [glp['QALYs_Add'].mean(), glp['QALYs_Mult'].mean()]
    
    ax2.bar(x_pos - width/2, base_means, width, label='Baseline', color='gray', alpha=0.5)
    ax2.bar(x_pos + width/2, glp_means, width, label='GLP-1 Intervention', color='seagreen')
    
    ax2.set_ylabel('Mean QALYs per Patient')
    ax2.set_title('Quality-Adjusted Life Years by Accounting Method', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Additive (DW Sum)', 'Multiplicative (Utility)'])
    ax2.legend()
    
    # FIX: Ensure y-axis starts at zero for absolute perspective
    ax2.set_ylim(bottom=0) 
    # Add a bit of headroom for labels
    ax2.set_ylim(top=max(base_means + glp_means) * 1.2)
    
    plt.tight_layout()
    plt.savefig("delfino_comprehensive_results.png")
    print("📈 Comprehensive plots saved to 'delfino_comprehensive_results.png'")

if __name__ == "__main__":
    plot()