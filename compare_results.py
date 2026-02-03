import pandas as pd
import os

def quick_compare():
    print("📊 Generating Comparison Summary from existing CSVs...")
    
    if not os.path.exists("delfino_individual_base.csv") or not os.path.exists("delfino_individual_glp1.csv"):
        print("❌ Error: Result files not found.")
        return

    base = pd.read_csv("delfino_individual_base.csv")
    glp1 = pd.read_csv("delfino_individual_glp1.csv")

    summary_data = []
    
    # 1. Economic Totals
    summary_data.append(["Total Cost", base['Cost'].sum(), glp1['Cost'].sum()])
    summary_data.append(["Total DALYs", base['DALYs'].sum(), glp1['DALYs'].sum()])

    # 2. Disease Incidence (Robust Union Approach)
    all_inc_cols = sorted(list(set([c for c in base.columns if c.startswith('inc_')]) | 
                               set([c for c in glp1.columns if c.startswith('inc_')])))

    for col in all_inc_cols:
        # Count cases (> 0 excludes -1 and -99)
        b_count = (base[col] > 0).sum() if col in base.columns else 0
        g_count = (glp1[col] > 0).sum() if col in glp1.columns else 0
        
        summary_data.append([f"Cases: {col[4:]}", b_count, g_count])

    df_summary = pd.DataFrame(summary_data, columns=["Metric", "Baseline", "Intervention"])
    df_summary["Delta"] = df_summary["Baseline"] - df_summary["Intervention"]
    
    df_summary.to_csv("delfino_comparison_summary.csv", index=False)
    print("="*60)
    print(df_summary.to_string(index=False))
    print("="*60)
    print(f"📁 Summary updated in delfino_comparison_summary.csv")

if __name__ == "__main__":
    quick_compare()