import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_and_plot(k=10, min_base_cases=5):
    # 1. Load Data
    base = pd.read_csv('delfino_individual_base.csv')
    glp = pd.read_csv('delfino_individual_glp1.csv')
    
    # Load Name Map (if available from previous prevalence checks)
    name_map = {}
    if os.path.exists('training_data_prevalence.csv'):
        prev_df = pd.read_csv('training_data_prevalence.csv')
        name_map = dict(zip(prev_df['Code'], prev_df['Name']))

    # NEW: Identify disease columns by excluding metrics
    metrics_cols = ['ID', 'Cost', 'YLD', 'YLL', 'DALYs', 'QALYs_Add', 'QALYs_Mult']
    inc_cols = [c for c in base.columns if c not in metrics_cols]
    
    # 2. Calculate Proportional Metrics
    impact_list = []
    for col in inc_cols:
        b_count = (base[col] > 0).sum()
        g_count = (glp[col] > 0).sum()
        if b_count >= min_base_cases:
            prop_red = (b_count - g_count) / b_count
            impact_list.append({
                'code': col, # No longer stripping 'inc_'
                'base': b_count,
                'glp': g_count,
                'prop_red': prop_red
            })
    
    impact_df = pd.DataFrame(impact_list)
    
    # Helper function for consistent multi-panel plotting
    def create_figure(subset, filename, title_prefix):
        num = len(subset)
        if num == 0: return
        rows = (num + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(16, 5 * rows))
        if rows == 1: axes = np.array([axes])
        axes = axes.flatten()
        
        for i, (_, row) in enumerate(subset.iterrows()):
            ax, code = axes[i], row['code']
            full_name = name_map.get(code, code)
            
            b_ages = base[base[code] > 0][code].sort_values()
            g_ages = glp[glp[code] > 0][code].sort_values()
            
            ax.step(b_ages, np.arange(1, len(b_ages) + 1), label='Baseline', color='gray', alpha=0.6)
            ax.step(g_ages, np.arange(1, len(g_ages) + 1), label='GLP-1', color='#2c7fb8', linewidth=2.5)
            
            ax.set_title(f"{full_name}\n({row['prop_red']:.1%} Reduction | N_base={row['base']})", fontsize=11, fontweight='bold')
            ax.set_xlabel("Age")
            ax.set_ylabel("Cumulative Cases")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)

        for j in range(i + 1, len(axes)): axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    # --- PLOT 1: TOP-K PROPORTIONAL IMPACT ---
    top_k_subset = impact_df.sort_values('prop_red', ascending=False).head(k)
    create_figure(top_k_subset, 'delfino_top_proportional.png', "Top Proportional")

    # --- PLOT 2: INTERVENTION TARGETS (GERRYMANDERED) ---
    target_codes = ['E10', 'I50', 'I21', 'I63', 'N18']
    targets_subset = impact_df[impact_df['code'].isin(target_codes)].copy()
    
    # Sort targets specifically in the order requested
    targets_subset['code'] = pd.Categorical(targets_subset['code'], categories=target_codes, ordered=True)
    targets_subset = targets_subset.sort_values('code')
    
    create_figure(targets_subset, 'delfino_intervention_targets.png', "Targeted Outcomes")

    print(f"✅ Generated 'delfino_top_proportional.png' (Top {k})")
    print(f"✅ Generated 'delfino_intervention_targets.png' (Gerrymandered targets)")

if __name__ == "__main__":
    analyze_and_plot(k=8, min_base_cases=5)