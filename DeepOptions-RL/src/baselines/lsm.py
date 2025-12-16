import numpy as np

def longstaff_schwartz(paths, K, r, T, poly_degree=2):
    """
    Longstaff-Schwartz Algorithm for American Option Pricing.
    """
    n_paths, n_steps_plus_1 = paths.shape
    dt = T / (n_steps_plus_1 - 1)
    
    cash_flows = np.maximum(K - paths[:, -1], 0) # Payoff at maturity
    discount_factor = np.exp(-r * dt)
    
    # Backward induction
    for t in range(n_steps_plus_1 - 2, 0, -1):
        cash_flows = cash_flows * discount_factor # Discount previous cashflows
        
        in_the_money = paths[:, t] < K
        X = paths[in_the_money, t]
        Y = cash_flows[in_the_money] 
        
        if len(X) > 0:
            # Polynomial Regression to estimate Continuation Value
            coeffs = np.polyfit(X, Y, poly_degree)
            continuation_value = np.polyval(coeffs, X)
            exercise_value = K - X
            
            # Identify optimal exercise decisions
            exercise_now = exercise_value > continuation_value
            
            # Update cash flows where exercise is optimal
            # Map back to full paths indices
            indices = np.where(in_the_money)[0]
            cash_flows[indices[exercise_now]] = exercise_value[exercise_now]
            
    return np.mean(cash_flows * np.exp(-r * dt)) # Discount back one last step to t=0