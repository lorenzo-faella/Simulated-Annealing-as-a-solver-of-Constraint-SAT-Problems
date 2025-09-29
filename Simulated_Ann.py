 
import numpy as np

def accept(delta_c, beta):
    """Stochastically determine whether to accept a move according to Metropolis rule"""
    if delta_c <= 0:
        return True
    if beta == np.inf:
        return False
    return np.random.rand() < np.exp(-beta * delta_c)

def simulated_annealing(problem,
                       anneal_steps=10, mcmc_steps=100,
                       beta0=0.1, beta1=10.0,
                       seed=None, debug_delta_cost=False,
                       early_stop=True,  # Better name than 'optimize'
                       track_acceptance=False,
                       track_solution_step=False,
                       verbose=False):

    if seed is not None:
        np.random.seed(seed)

    # Temperature schedule
    beta_list = np.zeros(anneal_steps)
    beta_list[:-1] = np.linspace(beta0, beta1, anneal_steps - 1)
    beta_list[-1] = np.inf

    # Initialize problem
    problem.init_config()
    current_cost = problem.cost()
    best_cost = current_cost
    
    if verbose:
        print(f"Initial cost: {current_cost}")

    # Tracking variables
    solution_found_step = None
    acceptance_rates = []
    
    # Main annealing loop
    for step, beta in enumerate(beta_list):
        accepted_moves = 0
        
        for t in range(mcmc_steps):
            move = problem.propose_move()
            delta_cost = problem.compute_delta_cost(move)
            
            # Debug: verify delta cost computation
            if debug_delta_cost:
                problem_copy = problem.copy()
                problem_copy.accept_move(move)
                expected_cost = current_cost + delta_cost
                actual_cost = problem_copy.cost()
                assert abs(expected_cost - actual_cost) < 1e-10, \
                    f"Delta cost mismatch: {expected_cost} vs {actual_cost}"
            
            # Metropolis acceptance
            if accept(delta_cost, beta):
                problem.accept_move(move)
                current_cost += delta_cost
                accepted_moves += 1
                
                # Update best solution
                if current_cost < best_cost:
                    best_cost = current_cost
                    
                    # Check if optimal solution found
                    if best_cost == 0 and early_stop:
                        solution_found_step = step
                        if verbose:
                            print(f"Optimal solution found at step {step}")
                        break
        
        # Record acceptance rate for this temperature
        acceptance_rate = accepted_moves / mcmc_steps
        if track_acceptance:
            acceptance_rates.append(acceptance_rate)
            
        if verbose:
            print(f"Step {step}: beta={beta:.3f}, cost={current_cost}, "
                  f"best={best_cost}, acc_rate={acceptance_rate:.3f}")
        
        # Early termination if solution found
        if best_cost == 0 and early_stop and solution_found_step is not None:
            break

    # Prepare return values based on requested tracking
    result = [best_cost]
    
    if track_acceptance:
        result.append(acceptance_rates)
    
    if track_solution_step:
        result.append(solution_found_step)
    
    return result[0] if len(result) == 1 else tuple(result)



# Create K-SAT problem
ksat = KSAT(N=100, M=200, K=3, seed=42)

# Run simulated annealing with tracking
best_cost, acceptance_rates, solution_step = simulated_annealing(
    ksat,
    anneal_steps=20,
    mcmc_steps=1000,
    beta0=0.1,
    beta1=5.0,
    early_stop=True,
    track_acceptance=True,
    track_solution_step=True,
    verbose=True
)

print(f"Best cost: {best_cost}")
print(f"Solution found at step: {solution_step}")
print(f"Acceptance rates: {acceptance_rates}")
# %%
