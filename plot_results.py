import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def get_code_to_name_map():
    # Load labels and params to create a mapping
    data_dir = os.path.join('data', 'ukb_simulated_data')
    with open(os.path.join(data_dir, 'labels.csv'), 'r') as f:
        labels_list = [line.strip() for line in f.readlines()]
    
    params = pd.read_csv('dummy_disease_params.csv')
    # Map Code -> Full Name (e.g., "I21" -> "I21 (acute myocardial infarction)")
    mapping = {row['Code']: labels_list[int(row['TokenID'])] for _, row in params.iterrows()}
    return mapping

def analyze_and_plot(k=10):
    if not os.path.exists('delfino_individual_base.csv'):
        print("Error: Merged CSVs not found.")
        return

    base = pd.read_csv('delfino_individual_base.csv')
    glp = pd.read_csv('delfino_individual_glp1.csv')
    name_map = get_code_to_name_map()

    inc_cols = [c for c in base.columns if c.startswith('inc_')]
    
    # Calculate Delta
    deltas = {col: (base[col] > 0).sum() - (glp[col] > 0).sum() for col in inc_cols}
    top_k_cols = sorted(deltas, key=lambda x: abs(deltas[x]), reverse=True)[:k]
    
    fig, axes = plt.subplots(nrows=(k+1)//2, ncols=2, figsize=(16, 5 * ((k+1)//2)))
    axes = axes.flatten()

    for i, col in enumerate(top_k_cols):
        ax = axes[i]
        code = col[4:] # strip 'inc_'
        full_name = name_map.get(code, code) # Fallback to code if name missing

        base_ages = base[base[col] > 0][col].sort_values()
        glp_ages = glp[glp[col] > 0][col].sort_values()

        ax.step(base_ages, np.arange(1, len(base_ages) + 1), label='Baseline', color='gray', alpha=0.6)
        ax.step(glp_ages, np.arange(1, len(glp_ages) + 1), label='GLP-1', color='#2c7fb8', linewidth=2.5)
        
        # Heading with both Code and Name
        ax.set_title(f"Impact on {full_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Age of Onset")
        ax.set_ylabel("Cumulative Cases")
        ax.legend()

    plt.tight_layout()
    plt.savefig('delfino_top_impact_named.png', dpi=300)
    print(f"📈 Named plots saved to 'delfino_top_impact_named.png'")

if __name__ == "__main__":
    analyze_and_plot(k=8)