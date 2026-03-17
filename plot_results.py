import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def analyze_and_plot(start_id=0, end_id=7143, k=10, min_base_cases=30):
    # 1. Load Data
    control_file = f"control_{start_id}_{end_id}_incidence.csv"
    treated_file = f"treated_{start_id}_{end_id}_incidence.csv"
    labels_path = os.path.join('data', 'ukb_simulated_data', 'labels.csv')

    if not os.path.exists(control_file) or not os.path.exists(treated_file):
        print(f"Error: Could not find {control_file} or {treated_file}")
        return

    control = pd.read_csv(control_file)
    treated = pd.read_csv(treated_file)

    # 2. Load Disease Names for titles
    name_map = {}
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
            # Create a map where 'I10' -> 'I10 (essential (primary) hypertension)'
            for label in labels:
                code = label.split(' ')[0]
                name_map[code] = label

    # Identify disease columns
    inc_cols = [c for c in control.columns if c not in ['PatientID', 'SimulationStartAge']]
    
    impact_list = []
    for col in inc_cols:
        c_count = (control[col] > 0).sum()
        t_count = (treated[col] > 0).sum()
        
        if c_count >= min_base_cases:
            prop_red = (c_count - t_count) / c_count
            impact_list.append({
                'code': col, 
                'control_n': c_count,
                'treated_n': t_count,
                'prop_red': prop_red
            })
    
    impact_df = pd.DataFrame(impact_list)
    
    def create_figure(subset, filename):
        if subset.empty:
            return

        n = len(subset)
        cols = 2
        rows = (n + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
        axes = axes.flatten()

        for i, (idx, row) in enumerate(subset.iterrows()):
            ax = axes[i]
            code = row['code']
            # Fetch full name (e.g., "I10 (essential (primary) hypertension)")
            display_name = name_map.get(code, code)

            c_events = control[control[code] > 0]
            c_years = sorted(c_events[code] - c_events['SimulationStartAge'])
            
            t_events = treated[treated[code] > 0]
            t_years = sorted(t_events[code] - t_events['SimulationStartAge'])

            ax.step([0] + c_years, np.arange(0, len(c_years) + 1), 
                    label='Control', color='#636363', linewidth=2)
            ax.step([0] + t_years, np.arange(0, len(t_years) + 1), 
                    label='Treated (GLP-1)', color='#2c7fb8', linewidth=2.5)
            
            # Updated Title: Removed 'ICD-10:' and used display_name
            ax.set_title(f"{display_name}\n({row['prop_red']:.1%} Reduction | N_ctrl={row['control_n']})", 
                         fontsize=12, fontweight='bold')
            ax.set_xlabel("Years of Follow-up")
            ax.set_ylabel("Cumulative New Cases")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Generated Plot: {filename}")
        plt.close()

    # Generate Top-K and Target plots
    top_k_subset = impact_df.sort_values('prop_red', ascending=False).head(k)
    create_figure(top_k_subset, f'impact_top_reduction_{start_id}_{end_id}.png')

    target_codes = ['E11', 'I50', 'I21', 'I63', 'N18', 'I10']
    targets_subset = impact_df[impact_df['code'].isin(target_codes)].copy()
    create_figure(targets_subset, f'impact_clinical_targets_{start_id}_{end_id}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--end_id', type=int, default=7143)
    args = parser.parse_args()

    analyze_and_plot(start_id=args.start_id, end_id=args.end_id)