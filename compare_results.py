import pandas as pd

def compare(start_id=0, end_id=150, strategy='always', trigger_codes=''):
    
    # Reconstruct filenames based on strategy
    if strategy == "always":
        treat_tag = "treated_always"
    else:
        safe_codes = trigger_codes.replace(",", "-")
        treat_tag = f"treated_{strategy}_{safe_codes}"

    control_file = f"control_{start_id}_{end_id}_incidence.csv"
    treated_file = f"{treat_tag}_{start_id}_{end_id}_incidence.csv"

    base = pd.read_csv(control_file)
    glp = pd.read_csv(treated_file)

    # metrics = ["Cost", "YLD", "YLL", "DALYs", "QALYs_Add", "QALYs_Mult"]
    # summary = []
    # for m in metrics:
    #     summary.append([m, base[m].mean(), glp[m].mean()])

    # df_sum = pd.DataFrame(summary, columns=["Metric", "Base", "GLP1"])
    # df_sum["Delta"] = df_sum["GLP1"] - df_sum["Base"]
    
    # df_sum.to_csv("delfino_comparison_summary.csv", index=False)

    # Use the exact column names from the new incidence files
    metrics = ["Total_Costs", "Total_QALYs", "Total_YLDs", "Total_YLLs", "Total_DALYs"]
    
    summary = []
    for m in metrics:
        summary.append([m, base[m].mean(), glp[m].mean()])

    df_sum = pd.DataFrame(summary, columns=["Metric", "Control", "Treated"])
    df_sum["Delta"] = df_sum["Treated"] - df_sum["Control"]
    
    # Save descriptive summary
    summary_name = f"summary_{treat_tag}_{start_id}_{end_id}.csv"
    df_sum.to_csv(summary_name, index=False)


    
    # q_gain = df_sum.loc[df_sum['Metric'] == 'QALYs_Mult', 'Delta'].values[0]
    # c_inc = df_sum.loc[df_sum['Metric'] == 'Cost', 'Delta'].values[0]
    
    # print("-" * 30)
    # print(df_sum.to_string(index=False))
    # if q_gain > 0:
    #     print(f"\n💰 ICER (Mult): £{c_inc / q_gain:,.2f} / QALY")
    # print("-" * 30)

    # ICER = (Cost_Treated - Cost_Control) / (QALY_Treated - Control)
    q_gain = df_sum.loc[df_sum['Metric'] == 'Total_QALYs', 'Delta'].values[0]
    c_inc = df_sum.loc[df_sum['Metric'] == 'Total_Costs', 'Delta'].values[0]
    
    print("-" * 30)
    print(f"Trial Results: {treat_tag} (n={end_id-start_id})")
    print(df_sum.to_string(index=False))
    
    if q_gain != 0:
        icer = c_inc / q_gain
        print(f"\n💰 ICER: £{icer:,.2f} per QALY gained")
    
    # DALY reduction (Burden avoided)
    daly_saved = -df_sum.loc[df_sum['Metric'] == 'Total_DALYs', 'Delta'].values[0]
    print(f"📉 Total Burden Avoided: {daly_saved:.4f} DALYs per patient")
    print("-" * 30)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--end_id', type=int, default=150)
    parser.add_argument('--strategy', type=str, default='always')
    parser.add_argument('--trigger_codes', type=str, default='')
    args = parser.parse_args()
    
    compare(
        start_id=args.start_id, 
        end_id=args.end_id, 
        strategy=args.strategy, 
        trigger_codes=args.trigger_codes
    )
    # compare()