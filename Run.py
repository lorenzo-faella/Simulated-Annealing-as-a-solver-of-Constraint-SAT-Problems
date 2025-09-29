#%%import SimAnn
import KSAT

def empirical_probability(N=200, M=None, K=3, iterations=100, 
                         mcmc_steps=200, anneal_steps=20, 
                         beta0=1, beta1=10, seed=None, verbose=False):

    if M is None:
        M = N  # Common to use M = N for random K-SAT
    
    solved_count = 0
    cost_history = []
    
    for i in range(iterations):
        # Create new random K-SAT instance for each iteration
        ksat = KSAT.KSAT(N=N, M=M, K=K, seed=seed if seed is None else seed + i)
        
        # Run simulated annealing
        best_cost = SimAnn.simann(
            ksat,
            mcmc_steps=mcmc_steps,
            anneal_steps=anneal_steps,
            beta0=beta0,
            beta1=beta1,
            seed=seed if seed is None else seed + i + 1000,
            optimize=True
        )
        
        # Check if solution was found
        if best_cost == 0:
            solved_count += 1
        
        cost_history.append(best_cost)
        
        if verbose and (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{iterations}: "
                  f"Current success rate = {solved_count / (i + 1):.3f}")
    
    empirical_prob = solved_count / iterations
    
    if verbose:
        print(f"\nFinal Results:")
        print(f"Success rate: {empirical_prob:.3f} ({solved_count}/{iterations})")
        print(f"Average best cost: {np.mean(cost_history):.3f}")
        print(f"Cost std dev: {np.std(cost_history):.3f}")
    
    return empirical_prob
# %%

#%%


import SimAnn
import importlib


seed = 42

def run3sat(N=200, M=200, mcmc_steps=100, anneal_steps=10, beta0=1, beta1=10, seed=seed, acceptance_rate=True):
        ksat = KSAT.KSAT(N, M, 3, seed = seed)
        
        res = SimAnn.simann(ksat,
                               mcmc_steps = mcmc_steps, anneal_steps = anneal_steps,
                               beta0 = beta0, beta1 = beta1,
                               seed = seed,
                               debug_delta_cost = False, # set to True to enable the check
                               acceptance_rate=acceptance_rate)
        return res

