"""
Interactive Dashboard 7: Secretary Problem Simulator
Topic: Optimal stopping and decision-making
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple, Dict, Any
import pandas as pd

def generate_applicant_pool(n: int) -> List[float]:
    """Generate a pool of n applicants with quality scores."""
    # Generate unique quality scores from 0 to 1
    scores = np.random.uniform(0, 1, n)
    return scores.tolist()

def simulate_secretary_problem(applicants: List[float], look_percentage: float) -> Dict[str, Any]:
    """Simulate the secretary problem with given parameters."""
    n = len(applicants)
    look_count = int(n * look_percentage / 100)
    
    # Shuffle applicants to randomize order
    shuffled = applicants.copy()
    np.random.shuffle(shuffled)
    
    # Look phase: find the best candidate in the first look_count applicants
    if look_count > 0:
        benchmark = max(shuffled[:look_count])
    else:
        benchmark = 0
    
    # Leap phase: hire the first candidate better than benchmark
    hired_score = None
    hired_position = None
    
    for i in range(look_count, n):
        if shuffled[i] > benchmark:
            hired_score = shuffled[i]
            hired_position = i + 1
            break
    
    # If no one was hired, hire the last person
    if hired_score is None:
        hired_score = shuffled[-1]
        hired_position = n
    
    # Determine if we hired the best candidate
    best_score = max(applicants)
    is_best = (hired_score == best_score)
    
    # Calculate rank (1 = best, n = worst)
    sorted_scores = sorted(applicants, reverse=True)
    rank = sorted_scores.index(hired_score) + 1
    
    return {
        'hired_score': hired_score,
        'hired_position': hired_position,
        'is_best': is_best,
        'rank': rank,
        'benchmark': benchmark,
        'best_score': best_score,
        'applicant_order': shuffled
    }

def run_multiple_simulations(n_applicants: int, look_percentage: float, n_simulations: int) -> List[Dict[str, Any]]:
    """Run multiple simulations and collect results."""
    results = []
    
    for _ in range(n_simulations):
        applicants = generate_applicant_pool(n_applicants)
        result = simulate_secretary_problem(applicants, look_percentage)
        results.append(result)
    
    return results

def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate statistics from simulation results."""
    if not results:
        return {}
    
    success_rate = sum(1 for r in results if r['is_best']) / len(results)
    avg_rank = np.mean([r['rank'] for r in results])
    avg_score = np.mean([r['hired_score'] for r in results])
    avg_position = np.mean([r['hired_position'] for r in results])
    
    return {
        'success_rate': success_rate,
        'avg_rank': avg_rank,
        'avg_score': avg_score,
        'avg_position': avg_position
    }

def create_performance_plot(look_percentages: List[float], success_rates: List[float], 
                          avg_ranks: List[float]) -> go.Figure:
    """Create performance plot showing success rate and average rank vs look percentage."""
    fig = go.Figure()
    
    # Success rate
    fig.add_trace(go.Scatter(
        x=look_percentages,
        y=success_rates,
        mode='lines+markers',
        name='Success Rate (Best Candidate)',
        yaxis='y',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))
    
    # Average rank (secondary y-axis)
    fig.add_trace(go.Scatter(
        x=look_percentages,
        y=avg_ranks,
        mode='lines+markers',
        name='Average Rank',
        yaxis='y2',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))
    
    # Optimal 37% line
    optimal_percentage = 100 / np.e  # approximately 36.8%
    fig.add_vline(x=optimal_percentage, line_dash="dash", line_color="green",
                  annotation_text="Optimal (37%)", annotation_position="top right")
    
    fig.update_layout(
        title="Performance vs Look Percentage",
        xaxis_title="Look Percentage (%)",
        yaxis=dict(title="Success Rate", side="left"),
        yaxis2=dict(title="Average Rank", side="right", overlaying="y"),
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_rank_distribution_plot(results: List[Dict[str, Any]]) -> go.Figure:
    """Create histogram showing distribution of hired candidate ranks."""
    ranks = [r['rank'] for r in results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=ranks,
        nbinsx=min(20, max(ranks)),
        name='Rank Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Distribution of Hired Candidate Ranks",
        xaxis_title="Rank (1 = Best)",
        yaxis_title="Frequency",
        showlegend=False
    )
    
    return fig

def main():
    st.title("👔 Secretary Problem Simulator")
    st.markdown("Explore optimal stopping strategies for hiring decisions")
    
    # Sidebar controls
    st.sidebar.header("Simulation Settings")
    
    n_applicants = st.sidebar.slider(
        "Number of Applicants:",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )
    
    look_percentage = st.sidebar.slider(
        "Look Percentage (%):",
        min_value=0.0,
        max_value=100.0,
        value=37.0,
        step=1.0
    )
    
    n_simulations = st.sidebar.slider(
        "Number of Simulations:",
        min_value=10,
        max_value=10000,
        value=1000,
        step=10
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Single Simulation")
        
        if st.button("Run Single Simulation", type="primary"):
            # Generate applicant pool
            applicants = generate_applicant_pool(n_applicants)
            
            # Run simulation
            result = simulate_secretary_problem(applicants, look_percentage)
            
            # Display results
            st.metric("Hired Score", f"{result['hired_score']:.4f}")
            st.metric("Best Score", f"{result['best_score']:.4f}")
            st.metric("Hired Position", result['hired_position'])
            st.metric("Rank", f"{result['rank']}/{n_applicants}")
            
            # Success indicator
            if result['is_best']:
                st.success("✅ Hired the best candidate!")
            else:
                st.error(f"❌ Did not hire the best candidate (rank {result['rank']})")
            
            # Show applicant order
            st.subheader("📋 Interview Order")
            df = pd.DataFrame({
                'Position': range(1, n_applicants + 1),
                'Score': result['applicant_order'],
                'Hired': ['Yes' if i == result['hired_position'] - 1 else 'No' 
                         for i in range(n_applicants)]
            })
            st.dataframe(df, use_container_width=True)
    
    with col2:
        st.subheader("📊 Multiple Simulations")
        
        if st.button("Run Multiple Simulations"):
            # Run simulations
            results = run_multiple_simulations(n_applicants, look_percentage, n_simulations)
            
            # Calculate statistics
            stats = calculate_statistics(results)
            
            # Display statistics
            st.metric("Success Rate", f"{stats['success_rate']:.1%}")
            st.metric("Average Rank", f"{stats['avg_rank']:.1f}")
            st.metric("Average Score", f"{stats['avg_score']:.4f}")
            st.metric("Average Position", f"{stats['avg_position']:.1f}")
            
            # Store results for comparison
            if 'comparison_results' not in st.session_state:
                st.session_state.comparison_results = []
            
            st.session_state.comparison_results.append({
                'look_percentage': look_percentage,
                'success_rate': stats['success_rate'],
                'avg_rank': stats['avg_rank']
            })
    
    # Performance comparison
    if 'comparison_results' in st.session_state and st.session_state.comparison_results:
        st.subheader("📈 Performance Comparison")
        
        # Extract data for plotting
        look_percentages = [r['look_percentage'] for r in st.session_state.comparison_results]
        success_rates = [r['success_rate'] for r in st.session_state.comparison_results]
        avg_ranks = [r['avg_rank'] for r in st.session_state.comparison_results]
        
        # Create performance plot
        perf_fig = create_performance_plot(look_percentages, success_rates, avg_ranks)
        st.plotly_chart(perf_fig, use_container_width=True)
        
        # Clear comparison button
        if st.button("Clear Comparison"):
            st.session_state.comparison_results = []
            st.rerun()
    
    # Theory section
    st.subheader("🧮 The 37% Rule")
    st.markdown("""
    **The Optimal Strategy:**
    1. **Look Phase:** Reject the first 37% of applicants (N/e where e ≈ 2.718)
    2. **Leap Phase:** Hire the first applicant better than the best seen in the look phase
    3. **Result:** ~37% chance of hiring the best candidate
    
    **Why 37%?**
    - Mathematically optimal for finding the single best candidate
    - Balances exploration (learning about quality distribution) vs exploitation (hiring)
    - The number 1/e ≈ 0.368 appears naturally in the mathematical solution
    
    **Key Insights:**
    - **Exploration:** First phase establishes a benchmark
    - **Exploitation:** Second phase commits to first good option
    - **Trade-off:** More exploration = better benchmark but fewer options left
    """)
    
    # Dynamic problem
    st.subheader("🧩 Dynamic Problem")
    st.markdown("""
    **Challenge:** What if you want to hire someone in the top 20% instead of the absolute best?
    
    **Your Task:**
    1. Run simulations with different look percentages (10%, 20%, 30%, 40%, 50%)
    2. Track success rate (hiring someone in top 20%)
    3. Find the optimal look percentage for this goal
    4. How does it compare to the 37% rule?
    
    **Variations to Explore:**
    - **Variant A:** You can recall one previously rejected candidate
    - **Variant B:** Maximize expected score instead of probability of best
    - **Variant C:** Different quality distributions (normal, exponential)
    
    **Real-World Applications:**
    - Dating and relationship decisions
    - Job searching strategies
    - House/apartment hunting
    - Investment timing
    """)
    
    # Learning objectives
    st.subheader("🎓 Learning Objectives")
    st.markdown("""
    **Key Concepts:**
    - **Optimal Stopping:** When to stop searching and commit
    - **Explore-Exploit Trade-off:** Learning vs acting
    - **Sequential Decision Making:** Decisions with irreversible consequences
    - **Mathematical Optimization:** Finding the best strategy
    
    **Decision-Making Framework:**
    1. **Define your objective** (best candidate vs good candidate)
    2. **Set your exploration budget** (how much time to spend learning)
    3. **Establish your threshold** (what constitutes "good enough")
    4. **Commit when you find it** (don't second-guess)
    
    **Beyond the Secretary Problem:**
    - Multi-armed bandits
    - A/B testing
    - Clinical trials
    - Resource allocation
    """)

if __name__ == "__main__":
    main()


