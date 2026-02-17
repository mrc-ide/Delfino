# ... (Imports and Setup same as v9.6) ...

# UPFRONT OPTIMIZATION: Now with three arrays
GLOBAL_DW = np.zeros(VOCAB_SIZE)
GLOBAL_UTILITY = np.ones(VOCAB_SIZE) # Default utility is 1.0 (perfect health)
GLOBAL_COSTS = np.zeros(VOCAB_SIZE)
IS_DISEASE = np.zeros(VOCAB_SIZE, dtype=bool)

disease_df = pd.read_csv('dummy_disease_params.csv')
for _, row in disease_df.iterrows():
    tid = int(row['TokenID'])
    GLOBAL_DW[tid] = row['DW']
    GLOBAL_UTILITY[tid] = row['Utility']
    GLOBAL_COSTS[tid] = row['Cost']
    IS_DISEASE[tid] = True

# ... (Helper functions same as before) ...

def simulate_patient(patient_id, apply_glp1):
    # ... (Initialization same as before) ...
    active_dw, active_utilities, active_costs = [], [], []
    
    # ... (History processing same as before) ...

    for year in range(args.time_horizon):
        # 1. DALY Calculation (YLD)
        total_yld += sum(active_dw)
        
        # 2. QALY Calculation (Additive)
        # Using 1 - sum(decrements) where decrement = (1 - utility)
        qalys_add += max(0, 1.0 - sum([1.0 - u for u in active_utilities]))
        
        # 3. QALY Calculation (Multiplicative)
        # Standard HRQoL approach: U_total = U1 * U2 * U3...
        u_mult = 1.0
        for u in active_utilities: u_mult *= u
        qalys_mult += u_mult
        
        total_costs += sum(active_costs)

        # ... (Inference and token sampling logic same as before) ...
        
        if IS_DISEASE[next_token] and not already_diagnosed[next_token]:
            active_dw.append(GLOBAL_DW[next_token])
            active_utilities.append(GLOBAL_UTILITY[next_token])
            active_costs.append(GLOBAL_COSTS[next_token])
            already_diagnosed[next_token] = True
            # ... (incidence recording) ...

    # ... (Return results) ...