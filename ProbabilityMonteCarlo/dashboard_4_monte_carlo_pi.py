"""
Interactive Dashboard 4: Monte Carlo π Estimator
Topic: Monte Carlo method fundamentals
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple
import math

def generate_random_points(n: int) -> Tuple[List[float], List[float]]:
    """Generate n random points in the square [-1, 1] x [-1, 1]."""
    x = np.random.uniform(-1, 1, n)
    y = np.random.uniform(-1, 1, n)
    return x.tolist(), y.tolist()

def check_inside_circle(x: float, y: float) -> bool:
    """Check if point (x,y) is inside the unit circle."""
    return x**2 + y**2 <= 1

def estimate_pi(x_coords: List[float], y_coords: List[float]) -> Tuple[float, int, int]:
    """Estimate π using Monte Carlo method."""
    n_total = len(x_coords)
    n_inside = sum(1 for x, y in zip(x_coords, y_coords) if check_inside_circle(x, y))
    pi_estimate = 4 * n_inside / n_total
    return pi_estimate, n_inside, n_total

def calculate_confidence_interval(pi_estimate: float, n_total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate confidence interval for π estimate."""
    # Standard error of the proportion
    p_hat = pi_estimate / 4  # proportion inside circle
    se = math.sqrt(p_hat * (1 - p_hat) / n_total)
    
    # Z-score for confidence level
    z_score = 1.96 if confidence == 0.95 else 1.645 if confidence == 0.90 else 2.576
    
    # Confidence interval for proportion
    margin = z_score * se
    ci_lower = 4 * (p_hat - margin)
    ci_upper = 4 * (p_hat + margin)
    
    return ci_lower, ci_upper

def create_scatter_plot(x_coords: List[float], y_coords: List[float], n_inside: int) -> go.Figure:
    """Create scatter plot showing points inside/outside circle."""
    fig = go.Figure()
    
    # Separate points inside and outside circle
    inside_x, inside_y = [], []
    outside_x, outside_y = [], []
    
    for x, y in zip(x_coords, y_coords):
        if check_inside_circle(x, y):
            inside_x.append(x)
            inside_y.append(y)
        else:
            outside_x.append(x)
            outside_y.append(y)
    
    # Add points inside circle
    if inside_x:
        fig.add_trace(go.Scatter(
            x=inside_x, y=inside_y,
            mode='markers',
            name='Inside Circle',
            marker=dict(color='red', size=4),
            hovertemplate='(%{x:.3f}, %{y:.3f})<br>Inside<extra></extra>'
        ))
    
    # Add points outside circle
    if outside_x:
        fig.add_trace(go.Scatter(
            x=outside_x, y=outside_y,
            mode='markers',
            name='Outside Circle',
            marker=dict(color='blue', size=4),
            hovertemplate='(%{x:.3f}, %{y:.3f})<br>Outside<extra></extra>'
        ))
    
    # Add circle boundary
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=circle_x, y=circle_y,
        mode='lines',
        name='Circle Boundary',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    
    # Add square boundary
    square_x = [-1, 1, 1, -1, -1]
    square_y = [-1, -1, 1, 1, -1]
    
    fig.add_trace(go.Scatter(
        x=square_x, y=square_y,
        mode='lines',
        name='Square Boundary',
        line=dict(color='gray', width=2, dash='dash'),
        showlegend=False
    ))
    
    fig.update_layout(
        title="Monte Carlo π Estimation",
        xaxis_title="X",
        yaxis_title="Y",
        xaxis=dict(range=[-1.1, 1.1]),
        yaxis=dict(range=[-1.1, 1.1]),
        showlegend=True,
        width=600,
        height=600
    )
    
    return fig

def create_convergence_plot(pi_estimates: List[float], sample_sizes: List[int]) -> go.Figure:
    """Create convergence plot showing π estimates vs sample size."""
    fig = go.Figure()
    
    # Add π estimates
    fig.add_trace(go.Scatter(
        x=sample_sizes,
        y=pi_estimates,
        mode='lines+markers',
        name='π Estimate',
        line=dict(color='blue'),
        marker=dict(size=6)
    ))
    
    # Add true π line
    fig.add_hline(y=math.pi, line_dash="dash", line_color="red", 
                  annotation_text="True π", annotation_position="top right")
    
    fig.update_layout(
        title="Convergence of π Estimate",
        xaxis_title="Number of Samples",
        yaxis_title="π Estimate",
        xaxis_type="log",
        showlegend=True
    )
    
    return fig

def main():
    st.title("🎯 Monte Carlo π Estimator")
    st.markdown("Estimate π using the classic dart-throwing simulation")
    
    # Sidebar controls
    st.sidebar.header("Simulation Settings")
    
    num_samples = st.sidebar.slider(
        "Number of Samples:",
        min_value=10,
        max_value=10000,
        value=1000,
        step=10
    )
    
    confidence_level = st.sidebar.selectbox(
        "Confidence Level:",
        [0.90, 0.95, 0.99],
        format_func=lambda x: f"{int(x*100)}%"
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎲 Dart Throwing Simulation")
        
        if st.button("Throw Darts!", type="primary"):
            # Generate random points
            x_coords, y_coords = generate_random_points(num_samples)
            
            # Estimate π
            pi_estimate, n_inside, n_total = estimate_pi(x_coords, y_coords)
            
            # Calculate confidence interval
            ci_lower, ci_upper = calculate_confidence_interval(pi_estimate, n_total, confidence_level)
            
            # Calculate error
            true_error = abs(pi_estimate - math.pi)
            
            # Store results in session state for convergence plot
            if 'pi_estimates' not in st.session_state:
                st.session_state.pi_estimates = []
                st.session_state.sample_sizes = []
            
            st.session_state.pi_estimates.append(pi_estimate)
            st.session_state.sample_sizes.append(num_samples)
            
            # Display results
            st.metric("π Estimate", f"{pi_estimate:.6f}")
            st.metric("True π", f"{math.pi:.6f}")
            st.metric("Absolute Error", f"{true_error:.6f}")
            st.metric("Points Inside Circle", f"{n_inside}/{n_total}")
            
            # Confidence interval
            st.write(f"**{int(confidence_level*100)}% Confidence Interval:** [{ci_lower:.6f}, {ci_upper:.6f}]")
            
            # Create scatter plot
            fig = create_scatter_plot(x_coords, y_coords, n_inside)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Statistics")
        
        if 'pi_estimates' in st.session_state and st.session_state.pi_estimates:
            st.write("**Recent Estimates:**")
            for i, (est, size) in enumerate(zip(st.session_state.pi_estimates[-5:], 
                                              st.session_state.sample_sizes[-5:])):
                error = abs(est - math.pi)
                st.write(f"n={size}: {est:.4f} (error: {error:.4f})")
            
            # Convergence plot
            if len(st.session_state.pi_estimates) > 1:
                st.subheader("📈 Convergence")
                conv_fig = create_convergence_plot(st.session_state.pi_estimates, 
                                                 st.session_state.sample_sizes)
                st.plotly_chart(conv_fig, use_container_width=True)
        
        # Clear results button
        if st.button("Clear Results"):
            st.session_state.pi_estimates = []
            st.session_state.sample_sizes = []
            st.rerun()
    
    # Theory section
    st.subheader("🧮 Theory Behind the Method")
    st.markdown("""
    **The Method:**
    1. We have a square with side length 2 (area = 4)
    2. Inside it, we have a circle with radius 1 (area = π)
    3. Ratio of areas: π/4
    4. If we throw darts randomly, the ratio of darts inside the circle 
       to total darts should approximate π/4
    5. Therefore: π ≈ 4 × (darts inside / total darts)
    
    **Why This Works:**
    - The law of large numbers ensures convergence
    - Error decreases as 1/√n where n is the number of samples
    - This is a classic example of Monte Carlo integration
    """)
    
    # Dynamic problem
    st.subheader("🧩 Dynamic Problem")
    st.markdown("""
    **Challenge:** How many samples do you need to estimate π to 3 decimal places?
    
    **Your Task:**
    1. Start with 100 samples and record the error
    2. Increase samples systematically (100, 500, 1000, 5000, 10000)
    3. Plot error vs sample size on log-log scale
    4. What pattern do you observe?
    5. Predict how many samples you'd need for 4 decimal places
    
    **Bonus:** Try estimating the area of other shapes (triangle, ellipse) using the same method!
    """)
    
    # Learning objectives
    st.subheader("🎓 Learning Objectives")
    st.markdown("""
    **Key Insights:**
    - **Random Sampling:** Can solve deterministic problems
    - **Convergence:** More samples = better accuracy
    - **Error Analysis:** Understanding uncertainty in estimates
    - **Monte Carlo Integration:** Using randomness for numerical integration
    
    **Real-World Applications:**
    - Financial risk assessment
    - Physics simulations
    - Computer graphics (ray tracing)
    - Optimization problems
    """)

if __name__ == "__main__":
    main()


