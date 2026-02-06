import pandas as pd
import os

def quick_compare():
    print("📊 Post-Processing: Generating Comparison Summary...")
    
    # 1. Check for source files
    if not os.path.exists("delfino_individual_base.csv") or not os.path.exists("delfino_individual_glp1.csv"):
        print("❌ Error: Missing individual result CSVs. Cannot compare.")
        return

    try:
        base = pd.read_csv("delfino_individual_base.csv")
        glp1 = pd.read_csv("delfino_individual_glp1.csv")
    except PermissionError:
        print("❌ Permission Error: Please close the individual CSV files in Excel!")
        return

    summary_data = []
    
    # 2. Economic Totals
    summary_data.append(["Total Cost", base['Cost'].sum(), glp1['Cost'].sum()])
    summary_data.append(["Total DALYs", base['DALYs'].sum(), glp1['DALYs'].sum()])

    # 3. Robust Disease Incidence Union
    # We find every 'inc_' column that exists in either file
    all_inc_cols = sorted(list(set([c for c in base.columns if c.startswith('inc_')]) | 
                               set([c for c in glp1.columns if c.startswith('inc_')])))

    for col in all_inc_cols:
        # At-Risk: Patients who didn't start with the disease (-99.0)
        # Note: If a column is missing in one file, we assume the at-risk count 
        # is the same as the other file (since they are twins)
        if col in base.columns:
            at_risk = (base[col] != -99.0).sum()
        else:
            at_risk = (glp1[col] != -99.0).sum()
            
        # New Cases: Recorded an age (> 0) during simulation
        b_new = (base[col] > 0).sum() if col in base.columns else 0
        g_new = (glp1[col] > 0).sum() if col in glp1.columns else 0
        
        # We only add to summary if there's at least one case or it's a priority disease
        if b_new > 0 or g_new > 0:
            summary_data.append([f"At-Risk: {col[4:]}", at_risk, at_risk])
            summary_data.append([f"New Cases: {col[4:]}", b_new, g_new])

    df_summary = pd.DataFrame(summary_data, columns=["Metric", "Baseline", "Intervention"])
    df_summary["Delta"] = df_summary["Baseline"] - df_summary["Intervention"]
    
    # 4. Final Save with Permission Catching
    try:
        df_summary.to_csv("delfino_comparison_summary.csv", index=False)
        print("="*60)
        print(df_summary.to_string(index=False))
        print("="*60)
        print(f"📁 Success! Comparison saved to delfino_comparison_summary.csv")
    except PermissionError:
        print("❌ CRITICAL: 'delfino_comparison_summary.csv' is still open in Excel!")
        print("Please close Excel and run this script again.")

if __name__ == "__main__":
    quick_compare()