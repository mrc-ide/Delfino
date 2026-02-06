import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_curves(start_age=40.0, horizon=40):
    print("📈 Generating Cumulative Incidence Plot with full names...")
    
    # 1. Load Data
    base = pd.read_csv("delfino_individual_base.csv")
    glp = pd.read_csv("delfino_individual_glp1.csv")
    
    # 2. Load Names Mapping
    params_path = "dummy_disease_params.csv"
    if os.path.exists(params_path):
        params_df = pd.read_csv(params_path)
        # Map Code (e.g. 'E11') -> Name (e.g. 'Non-insulin-dependent diabetes mellitus')
        name_map = dict(zip(params_df['Code'], params_df['Name']))
    else:
        name_map = {}

    # 3. Identify top 5 most frequent diseases (excluding death)
    inc_cols = [c for c in base.columns if c.startswith("inc_") and c != "inc_death"]
    counts = {c: (base[c] > 0).sum() for c in inc_cols}
    top_diseases = sorted(counts, key=counts.get, reverse=True)[:5]
    
    ages = np.arange(start_age, start_age + horizon + 1)
    plt.figure(figsize=(14, 9))

    for i, disease in enumerate(top_diseases):
        code = disease[4:] # Strip 'inc_' to get the ICD-10 code
        full_name = name_map.get(code, code) # Fallback to code if name not found
        
        # Calculate cumulative cases at each age
        b_curve = [(base[disease] <= age).where(base[disease] > 0).sum() for age in ages]
        g_curve = [(glp[disease] <= age).where(glp[disease] > 0).sum() if disease in glp.columns else 0 for age in ages]
        
        plt.plot(ages, b_curve, label=f"{full_name} (Base)", linestyle='--', color=f"C{i}", alpha=0.6)
        plt.plot(ages, g_curve, label=f"{full_name} (GLP-1)", linestyle='-', color=f"C{i}", linewidth=2.5)

    plt.title(f"Cumulative Disease Incidence (N={len(base)}, T={horizon})", fontsize=16)
    plt.xlabel("Patient Age", fontsize=12)
    plt.ylabel("Cumulative Number of Cases", fontsize=12)
    
    # Place legend outside the plot for readability given the long names
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("delfino_incidence_plot.png")
    print("✅ Plot saved as delfino_incidence_plot.png")

if __name__ == "__main__":
    plot_curves()