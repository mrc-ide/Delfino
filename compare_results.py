import pandas as pd

def compare():
    base = pd.read_csv("delfino_individual_base.csv")
    glp = pd.read_csv("delfino_individual_glp1.csv")

    metrics = ["Cost", "DALYs", "QALYs_Add", "QALYs_Mult"]
    summary = []
    for m in metrics:
        summary.append([m, base[m].mean(), glp[m].mean()])

    df_sum = pd.DataFrame(summary, columns=["Metric", "Base", "GLP1"])
    df_sum["Delta"] = df_sum["GLP1"] - df_sum["Base"]
    
    print("--- Population Means ---")
    print(df_sum.to_string(index=False))

    cost_delta = df_sum.loc[df_sum['Metric'] == 'Cost', 'Delta'].values[0]
    
    for q_type in ["QALYs_Add", "QALYs_Mult"]:
        gain = df_sum.loc[df_sum['Metric'] == q_type, 'Delta'].values[0]
        if gain > 0:
            print(f"💰 ICER ({q_type}): £{cost_delta / gain:,.2f} / QALY")
        else:
            print(f"💰 ICER ({q_type}): Intervention was not cost-effective (No gain).")

if __name__ == "__main__":
    compare()