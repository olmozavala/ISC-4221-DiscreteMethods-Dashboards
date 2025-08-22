# Interactive Dashboard Summary Table

## Complete Dashboard Proposals

| Topic | Dashboard Idea | Student Action | Key Insight |
|-------|----------------|----------------|-------------|
| **Sample Spaces & Events** | Probability Building Blocks Explorer | Adjust experiment type (dice/coins/cards), define custom events, run simulations | Understanding probability through hands-on experimentation |
| **Event Relationships** | Event Relationships Visualizer | Manipulate interactive Venn diagrams, toggle independence/dependence, calculate conditional probabilities | How events relate and how conditional information changes probabilities |
| **Random Number Generation** | RNG Laboratory | Compare PRNG vs TRNG, test different algorithms, analyze randomness quality | Difference between true and pseudo-random numbers and their applications |
| **Monte Carlo π Estimation** | Monte Carlo π Estimator | Throw virtual darts, adjust sample sizes, analyze convergence | Random sampling can solve deterministic problems with quantifiable accuracy |
| **Monte Carlo Integration** | Monte Carlo Integration Explorer | Select functions, adjust dimensions, compare with analytical solutions | Monte Carlo's power in high-dimensional problems (curse of dimensionality) |
| **Brownian Motion** | Brownian Motion Simulator | Adjust drift/volatility, track multiple particles, simulate stock prices | Random walks model natural and financial processes with mathematical precision |
| **Secretary Problem** | Secretary Problem Simulator | Test different look percentages, compare strategies, explore variants | Optimal stopping balances exploration and exploitation in sequential decisions |
| **Central Limit Theorem** | CLT Demonstrator | Vary sample sizes, compare distributions, observe convergence | Sample means converge to normal distribution regardless of original distribution |

## Detailed Dashboard Specifications

### Dashboard 1: Probability Building Blocks Explorer
- **Interactive Elements:** Experiment type selector, number of items slider, event definition, real-time simulation
- **Learning Goal:** Master fundamental probability concepts through experimentation
- **Dynamic Problem:** Design experiment with 3 dice where sum is 10-15, verify with 1000 trials

### Dashboard 2: Event Relationships Visualizer  
- **Interactive Elements:** Draggable Venn diagrams, probability calculators, independence toggles
- **Learning Goal:** Understand event relationships and conditional probability
- **Dynamic Problem:** Create events A and B where P(A∪B) = 0.8, find P(A∩B)

### Dashboard 3: Random Number Generator Laboratory
- **Interactive Elements:** Algorithm comparison, seed input, statistical tests, speed analysis
- **Learning Goal:** Distinguish between true and pseudo-random numbers
- **Dynamic Problem:** Design a "bad" PRNG and show why it fails tests, then fix it

### Dashboard 4: Monte Carlo π Estimator
- **Interactive Elements:** Dart-throwing simulation, sample size controls, confidence intervals
- **Learning Goal:** Master core Monte Carlo principles through π estimation
- **Dynamic Problem:** Find samples needed for π to 3 decimal places, analyze convergence

### Dashboard 5: Monte Carlo Integration Explorer
- **Interactive Elements:** Function selector, dimension controls, comparison plots
- **Learning Goal:** Understand Monte Carlo integration and curse of dimensionality
- **Dynamic Problem:** Compare Monte Carlo vs trapezoidal rule, find dimension where MC becomes more efficient

### Dashboard 6: Brownian Motion Simulator
- **Interactive Elements:** Drift/volatility controls, multi-particle tracking, stock price simulation
- **Learning Goal:** Explore stochastic processes and their applications
- **Dynamic Problem:** Simulate stock with μ=0.1, σ=0.2, find probability of doubling in 1 year

### Dashboard 7: Secretary Problem Simulator
- **Interactive Elements:** Look percentage slider, strategy comparison, performance analysis
- **Learning Goal:** Understand optimal stopping and explore-exploit tradeoffs
- **Dynamic Problem:** Find optimal look percentage for hiring top 20% instead of best

### Dashboard 8: Central Limit Theorem Demonstrator
- **Interactive Elements:** Distribution selector, sample size controls, convergence animation
- **Learning Goal:** Understand why Monte Carlo methods work through CLT
- **Dynamic Problem:** Find samples needed for mean distribution to look normal for different distributions

## Implementation Status

✅ **Completed Dashboards:**
- Dashboard 1: Probability Building Blocks Explorer
- Dashboard 4: Monte Carlo π Estimator  
- Dashboard 6: Brownian Motion Simulator
- Dashboard 7: Secretary Problem Simulator

🔄 **Remaining Dashboards to Implement:**
- Dashboard 2: Event Relationships Visualizer
- Dashboard 3: Random Number Generator Laboratory
- Dashboard 5: Monte Carlo Integration Explorer
- Dashboard 8: Central Limit Theorem Demonstrator

## Pedagogical Framework

### Scaffolding Approach
1. **Introduction:** Demonstrate with default settings
2. **Guided Exploration:** Start simple, gradually increase complexity
3. **Student Investigation:** Present dynamic problems for independent exploration
4. **Synthesis:** Connect back to theoretical concepts

### Assessment Integration
- **Formative:** Real-time feedback during exploration
- **Summative:** Dashboard outputs for homework/projects
- **Extension:** Student modifications and custom problems

### Real-World Connections
- **Finance:** Risk assessment, option pricing
- **Physics:** Particle diffusion, heat transfer
- **Computer Science:** Algorithm design, cryptography
- **Biology:** Population dynamics, molecular motion


