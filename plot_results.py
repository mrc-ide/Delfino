import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_all():
    print("📊 Loading results for plotting...")
    base = pd.read_csv("delfino_individual_base.csv")
    glp = pd.read_csv("delfino_individual_glp1.csv")
    params_df = pd.read_csv("dummy_disease_params.csv")
    
    # Create Name Lookup: Code -> Name
    name_map = dict(zip(params_df['Code'], params_df['Name']))

    # --- PLOT 1: CUMULATIVE INCIDENCE (TOP 5) ---
    inc_cols = [c for c in base.columns if c.startswith("inc_") and c != "inc_death"]
    # Get top 5 by count in Baseline
    counts = {c: (base[c] > 0).sum() for c in inc_cols}
    top_diseases = sorted(counts, key=counts.get, reverse=True)[:5]
    
    start_age, horizon = 40, 40
    ages = np.arange(start_age, start_age + horizon + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    for i, disease in enumerate(top_diseases):
        code = disease[4:] # Strip 'inc_'
        name = name_map.get(code, code)
        
        b_curve = [(base[disease] <= age).where(base[disease] > 0).sum() for age in ages]
        g_curve = [(glp[disease] <= age).where(glp[disease] > 0).sum() if disease in glp.columns else 0 for age in ages]
        
        ax1.plot(ages, b_curve, label=f"{name} (Base)", linestyle='--', color=f"C{i}", alpha=0.5)
        ax1.plot(ages, g_curve, label=f"{name} (GLP-1)", linestyle='-', color=f"C{i}", linewidth=2)

    ax1.set_title(f"Cumulative Disease Incidence (N={len(base)})", fontsize=14)
    ax1.set_ylabel("Total Number of Cases")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- PLOT 2: QALY COMPARISON (ADDITIVE vs MULTIPLICATIVE) ---
    qaly_metrics = ["QALYs_Add", "QALYs_Mult"]
    means_base = [base[m].mean() for m in qaly_metrics]
    means_glp = [glp[m].mean() for m in qaly_metrics]
    
    x = np.arange(len(qaly_metrics))
    width = 0.35

    ax2.bar(x - width/2, means_base, width, label='Baseline', color='gray', alpha=0.6)
    ax2.bar(x + width/2, means_glp, width, label='GLP-1 Intervention', color='seagreen')

    ax2.set_ylabel('Mean QALYs per Patient')
    ax2.set_title('Comparison of Quality-Adjusted Life Years by Accounting Method')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Additive (DW Sum)', 'Multiplicative (Utility Product)'])
    ax2.legend()
    ax2.set_ylim(min(means_base + means_glp) * 0.9, max(means_base + means_glp) * 1.1)

    plt.tight_layout()
    plt.savefig("delfino_comprehensive_results.png")
    print("✅ Comprehensive plot saved as delfino_comprehensive_results.png")

if __name__ == "__main__":
    plot_all()