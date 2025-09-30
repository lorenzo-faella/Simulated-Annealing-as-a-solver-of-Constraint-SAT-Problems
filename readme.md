# Simulated Annealing for K-SAT Problems

This project implements a Simulated Annealing solver for K-SAT constraint satisfaction problems, specifically focusing on 3-SAT instances. The code investigates how the satisfiability of random K-SAT instances depends on the relationship between the number of variables (N) and clauses (M).

## What's in this repository

### Core Files
- **KSAT.py** - Main class that defines the K-SAT problem with methods for cost calculation, move proposals, and delta cost computation
- **Simulated_Ann.py** - Simulated Annealing implementation with Metropolis acceptance rule and temperature scheduling
- **Run.py** - Scripts for empirical analysis of 3-SAT properties and algorithmic thresholds

### What the code does

The K-SAT class creates random SAT instances where:
- N binary variables need to satisfy M clauses
- Each clause contains exactly K literals (variables or their negations)
- The cost function counts unsatisfied clauses using an efficient vectorized approach
- Delta cost computation only recalculates affected clauses for performance

The Simulated Annealing solver:
- Uses a geometric temperature schedule from beta0 to beta1
- Implements symmetric move proposals (variable flips)
- Tracks acceptance rates and can stop early when solutions are found

### Research questions explored

1. **Acceptance rate evolution** - How does the acceptance probability change during annealing for fixed N=200, M=200?

2. **Scalability with clause density** - Can SA always find solutions as M increases to 400, 600, 800, 1000 with fixed N=200?

3. **Algorithmic threshold** - What's the empirical probability P(M,N) of solving random 3-SAT instances?
   - Plotted P(M,N) vs M for fixed N=200 across M ∈ {400,500,600,700,800,900,1000}
   - Identified the algorithmic threshold M_alg where P(M,N) = 1/2
   - Investigated scaling of M_alg with N for N ∈ {300,400,500,600}

### Key implementation details

- Cost function uses the formula: E(𝑥̂) = Σ_m Π_k (1 - s_mk * 𝑥̂_mk)/2
- Efficient delta cost computation by tracking which clauses contain each variable
- Vectorized operations using numpy for better performance
- Configurable annealing parameters (steps, temperatures, early stopping)

### Usage example

```python
# Create a 3-SAT instance
ksat = KSAT(N=200, M=200, K=3, seed=42)

# Run simulated annealing
best_cost = SimAnn.simann(ksat, anneal_steps=20, mcmc_steps=1000, 
                         beta0=0.1, beta1=10.0, optimize=True)

# Check if solution found
if best_cost == 0:
    print("SAT instance solved!")
