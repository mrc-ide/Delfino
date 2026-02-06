import pandas as pd

def compare():
    base = pd.read_csv("delfino_individual_base.csv")
    glp = pd.read_csv("delfino_individual_glp1.csv")

    metrics = ["Cost", "YLD", "YLL", "DALYs", "QALYs"]
    summary = []
    for m in metrics:
        summary.append([m, base[m].mean(), glp[m].mean()])

    df_sum = pd.DataFrame(summary, columns=["Metric", "Mean Baseline", "Mean Intervention"])
    df_sum["Delta (Gain/Loss)"] = df_sum["Mean Intervention"] - df_sum["Mean Baseline"]
    
    # Calculate ICER
    qaly_gain = df_sum.loc[df_sum['Metric'] == 'QALYs', 'Delta (Gain/Loss)'].values[0]
    cost_inc = df_sum.loc[df_sum['Metric'] == 'Cost', 'Delta (Gain/Loss)'].values[0]
    
    print(df_sum.to_string(index=False))
    if qaly_gain > 0:
        print(f"\n💰 ICER: £{cost_inc / qaly_gain:,.2f} per QALY gained")
    else:
        print("\n💰 No QALY gain detected to calculate ICER.")

if __name__ == "__main__":
    compare()