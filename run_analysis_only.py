import pandas as pd
import glob
from plot_results import analyze_and_plot

def quick_refresh():
    print("🔄 Refreshing merged data from temp files...")
    for group in ['base', 'glp1']:
        files = glob.glob(f"temp_{group}_*.csv")
        if files:
            pd.concat([pd.read_csv(f) for f in files]).to_csv(f'delfino_individual_{group}.csv', index=False)
    
    print("🎨 Re-generating named plots...")
    analyze_and_plot(k=10)

if __name__ == "__main__":
    quick_refresh()