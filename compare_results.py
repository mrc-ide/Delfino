import pandas as pd

def compare():
    base = pd.read_csv("delfino_individual_base.csv")
    glp = pd.read_csv("delfino_individual_glp1.csv")

    metrics = ["Cost", "YLD", "YLL", "DALYs", "QALYs_Add", "QALYs_Mult"]
    summary = []
    for m in metrics:
        summary.append([m, base[m].mean(), glp[m].mean()])

    df_sum = pd.DataFrame(summary, columns=["Metric", "Base", "GLP1"])
    df_sum["Delta"] = df_sum["GLP1"] - df_sum["Base"]
    
    df_sum.to_csv("delfino_comparison_summary.csv", index=False)
    
    q_gain = df_sum.loc[df_sum['Metric'] == 'QALYs_Mult', 'Delta'].values[0]
    c_inc = df_sum.loc[df_sum['Metric'] == 'Cost', 'Delta'].values[0]
    
    print("-" * 30)
    print(df_sum.to_string(index=False))
    if q_gain > 0:
        print(f"\n💰 ICER (Mult): £{c_inc / q_gain:,.2f} / QALY")
    print("-" * 30)

if __name__ == "__main__":
    compare()