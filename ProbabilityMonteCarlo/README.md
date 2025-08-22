# Interactive Probability and Monte Carlo Dashboards

This repository contains a collection of interactive Python dashboards designed to make probability theory and Monte Carlo methods engaging and accessible for students. Each dashboard focuses on specific concepts from the tutorial "A Student's Guide to Randomness and Computation."

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- `uv` package manager (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ProbabilityMonteCarlo
   ```

2. **Create and activate virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Run any dashboard:**
   ```bash
   streamlit run dashboard_1_probability_building_blocks.py
   ```

## 📊 Available Dashboards

### 1. Probability Building Blocks Explorer
**File:** `dashboard_1_probability_building_blocks.py`
**Topic:** Sample spaces, events, and random variables

**Features:**
- Interactive experiment selection (dice, coins, cards)
- Real-time sample space visualization
- Event probability calculation
- Frequency distribution analysis

**Learning Goal:** Understand fundamental probability concepts through hands-on experimentation.

### 2. Monte Carlo π Estimator
**File:** `dashboard_4_monte_carlo_pi.py`
**Topic:** Monte Carlo method fundamentals

**Features:**
- Interactive dart-throwing simulation
- Real-time π estimation with confidence intervals
- Convergence analysis
- Error visualization

**Learning Goal:** Master the core principle of using random sampling to estimate quantities.

### 3. Secretary Problem Simulator
**File:** `dashboard_7_secretary_problem.py`
**Topic:** Optimal stopping and decision-making

**Features:**
- Interactive hiring simulation
- Performance comparison across strategies
- 37% rule demonstration
- Explore-exploit tradeoff visualization

**Learning Goal:** Understand optimal stopping strategies and sequential decision-making.

### 4. Brownian Motion Simulator
**File:** `dashboard_6_brownian_motion.py`
**Topic:** Stochastic processes and random walks

**Features:**
- 2D Brownian motion visualization
- Stock price simulation (Geometric Brownian Motion)
- Multi-particle tracking
- Distance analysis

**Learning Goal:** Explore random walks and their applications in physics and finance.

## 🎯 How to Use in Class

### Before Class
1. **Setup:** Ensure all dashboards are working on your system
2. **Preparation:** Review the learning objectives for each dashboard
3. **Customization:** Modify parameters or add new features as needed

### During Class

#### Introduction (5-10 minutes)
- **Demonstrate:** Show the dashboard with default settings
- **Explain:** Connect to theoretical concepts from the tutorial
- **Motivate:** Explain why this topic matters in real-world applications

#### Guided Exploration (15-20 minutes)
- **Scaffold:** Start with simple parameters and gradually increase complexity
- **Question:** Ask students to predict outcomes before running simulations
- **Discuss:** Compare theoretical vs empirical results

#### Student Investigation (20-30 minutes)
- **Challenge:** Present the dynamic problem from each dashboard
- **Explore:** Let students experiment with different parameters
- **Analyze:** Guide them to discover patterns and insights

#### Synthesis (10-15 minutes)
- **Summarize:** Key takeaways from the exploration
- **Connect:** Link back to theoretical concepts
- **Extend:** Discuss real-world applications

### After Class
- **Assignment:** Students can continue exploring at home
- **Extension:** Challenge students to modify or extend the dashboards
- **Assessment:** Use dashboard outputs for homework or projects

## 🧩 Dynamic Problems for Each Dashboard

### Dashboard 1: Probability Building Blocks
**Problem:** Design an experiment with 3 dice where the sum is between 10-15. What's the probability of this event? Verify your calculation by running 1000 trials.

**Student Actions:**
- Set up experiment parameters
- Define custom events
- Run simulations
- Compare theoretical vs empirical probabilities

### Dashboard 4: Monte Carlo π Estimator
**Problem:** How many samples do you need to estimate π to 3 decimal places? Design an experiment to find out.

**Student Actions:**
- Run simulations with different sample sizes
- Record errors and convergence rates
- Plot error vs sample size
- Predict requirements for higher precision

### Dashboard 7: Secretary Problem
**Problem:** What if you want to hire someone in the top 20% instead of the absolute best? Find the optimal look percentage for this goal.

**Student Actions:**
- Test different look percentages
- Track success rates for modified objectives
- Compare strategies
- Explore variant problems

### Dashboard 6: Brownian Motion
**Problem:** Simulate a stock price with μ=0.1 and σ=0.2. What's the probability it doubles in 1 year? Verify with 1000 simulations.

**Student Actions:**
- Set up GBM parameters
- Run multiple simulations
- Calculate doubling probabilities
- Compare with theoretical expectations

## 🎓 Learning Objectives Summary

| Dashboard | Topic | Key Concepts | Student Actions | Key Insights |
|-----------|-------|--------------|-----------------|--------------|
| **1** | Probability Building Blocks | Sample spaces, events, random variables | Experiment with different setups, define events, run simulations | Understanding probability through experimentation |
| **4** | Monte Carlo π Estimation | Random sampling, convergence, error analysis | Vary sample sizes, track convergence, analyze errors | Random sampling can solve deterministic problems |
| **7** | Secretary Problem | Optimal stopping, explore-exploit tradeoff | Test strategies, compare performance, explore variants | Balancing exploration and exploitation in decisions |
| **6** | Brownian Motion | Stochastic processes, random walks, financial modeling | Simulate paths, analyze statistics, explore applications | Randomness in natural and financial systems |

## 🔧 Customization and Extension

### Adding New Features
Each dashboard is modular and can be easily extended:

```python
# Example: Add new experiment type to Dashboard 1
def calculate_sample_space(experiment_type: str, num_items: int) -> Dict[str, Any]:
    if experiment_type == "New Experiment":
        # Add your custom logic here
        pass
```

### Creating New Dashboards
Use the existing dashboards as templates:

1. **Copy structure** from an existing dashboard
2. **Modify functions** for your specific topic
3. **Add interactive elements** using Streamlit widgets
4. **Include learning objectives** and dynamic problems

### Integration with Other Tools
- **Jupyter Notebooks:** Export dashboard code to notebooks
- **Grading:** Use dashboard outputs for automated assessment
- **Collaboration:** Share dashboards via Streamlit Cloud

## 📚 Additional Resources

### Related Tutorial Sections
- **Part 1:** Fundamentals of Probability
- **Part 2:** Random Numbers and Distributions  
- **Part 3:** Monte Carlo Method
- **Part 4:** Stochastic Processes

### External Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [NumPy Documentation](https://numpy.org/doc/)

### Further Reading
- "Introduction to Probability Models" by Sheldon Ross
- "Monte Carlo Methods in Financial Engineering" by Paul Glasserman
- "Stochastic Processes" by Sheldon Ross

## 🤝 Contributing

We welcome contributions to improve these dashboards:

1. **Bug fixes:** Report issues and submit fixes
2. **New features:** Add new interactive elements
3. **Documentation:** Improve explanations and examples
4. **New dashboards:** Create dashboards for additional topics

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Streamlit Team** for the excellent web app framework
- **Plotly Team** for interactive visualization tools
- **NumPy/SciPy Teams** for scientific computing capabilities
- **Students and Instructors** who provide feedback and suggestions

---

**Happy Learning! 🎓✨**


