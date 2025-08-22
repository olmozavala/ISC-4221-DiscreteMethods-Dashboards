"""
Interactive Dashboard 6: Brownian Motion Simulator
Topic: Stochastic processes and random walks
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple, Dict
import pandas as pd

def simulate_brownian_motion(n_steps: int, dt: float, drift: float = 0.0, 
                           volatility: float = 1.0, n_particles: int = 1) -> Dict[str, List]:
    """Simulate Brownian motion for multiple particles."""
    # Time array
    time = np.linspace(0, n_steps * dt, n_steps + 1)
    
    # Initialize positions
    x_positions = np.zeros((n_particles, n_steps + 1))
    y_positions = np.zeros((n_particles, n_steps + 1))
    
    # Simulate each particle
    for particle in range(n_particles):
        # Generate random increments
        dx = np.random.normal(drift * dt, volatility * np.sqrt(dt), n_steps)
        dy = np.random.normal(drift * dt, volatility * np.sqrt(dt), n_steps)
        
        # Cumulative sum to get positions
        x_positions[particle, 1:] = np.cumsum(dx)
        y_positions[particle, 1:] = np.cumsum(dy)
    
    return {
        'time': time.tolist(),
        'x_positions': x_positions.tolist(),
        'y_positions': y_positions.tolist()
    }

def simulate_geometric_brownian_motion(n_steps: int, dt: float, initial_price: float = 100.0,
                                     drift: float = 0.05, volatility: float = 0.2) -> Dict[str, List]:
    """Simulate geometric Brownian motion for stock prices."""
    time = np.linspace(0, n_steps * dt, n_steps + 1)
    
    # Generate price path
    prices = [initial_price]
    
    for i in range(n_steps):
        # Random increment
        dW = np.random.normal(0, np.sqrt(dt))
        
        # Geometric Brownian motion equation
        # dS = S * (μ*dt + σ*dW)
        price_change = prices[-1] * (drift * dt + volatility * dW)
        new_price = prices[-1] + price_change
        
        prices.append(new_price)
    
    return {
        'time': time.tolist(),
        'prices': prices
    }

def calculate_statistics(positions: List[List[float]], time: List[float]) -> Dict[str, float]:
    """Calculate statistics for Brownian motion paths."""
    # Calculate distances from origin
    distances = []
    for particle_positions in positions:
        x_pos = particle_positions[0]
        y_pos = particle_positions[1]
        final_distance = np.sqrt(x_pos[-1]**2 + y_pos[-1]**2)
        distances.append(final_distance)
    
    # Calculate mean distance and theoretical expectation
    mean_distance = np.mean(distances)
    theoretical_distance = np.sqrt(2 * volatility * time[-1])  # For 2D Brownian motion
    
    return {
        'mean_distance': mean_distance,
        'theoretical_distance': theoretical_distance,
        'std_distance': np.std(distances)
    }

def create_path_plot(simulation_data: Dict[str, List], show_multiple: bool = True) -> go.Figure:
    """Create 2D path plot for Brownian motion."""
    fig = go.Figure()
    
    time = simulation_data['time']
    x_positions = simulation_data['x_positions']
    y_positions = simulation_data['y_positions']
    
    if show_multiple and len(x_positions) > 1:
        # Multiple particles
        colors = px.colors.qualitative.Set1
        for i, (x_pos, y_pos) in enumerate(zip(x_positions, y_positions)):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=x_pos, y=y_pos,
                mode='lines',
                name=f'Particle {i+1}',
                line=dict(color=color, width=2),
                hovertemplate='Time: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
                text=[f'{t:.2f}' for t in time]
            ))
            
            # Mark start and end points
            fig.add_trace(go.Scatter(
                x=[x_pos[0]], y=[y_pos[0]],
                mode='markers',
                marker=dict(color=color, size=8, symbol='circle'),
                showlegend=False,
                hovertemplate='Start<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=[x_pos[-1]], y=[y_pos[-1]],
                mode='markers',
                marker=dict(color=color, size=8, symbol='diamond'),
                showlegend=False,
                hovertemplate='End<extra></extra>'
            ))
    else:
        # Single particle
        x_pos = x_positions[0]
        y_pos = y_positions[0]
        
        fig.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='lines',
            name='Particle Path',
            line=dict(color='blue', width=3),
            hovertemplate='Time: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
            text=[f'{t:.2f}' for t in time]
        ))
        
        # Mark start and end points
        fig.add_trace(go.Scatter(
            x=[x_pos[0]], y=[y_pos[0]],
            mode='markers',
            marker=dict(color='green', size=10, symbol='circle'),
            name='Start',
            hovertemplate='Start<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[x_pos[-1]], y=[y_pos[-1]],
            mode='markers',
            marker=dict(color='red', size=10, symbol='diamond'),
            name='End',
            hovertemplate='End<extra></extra>'
        ))
    
    fig.update_layout(
        title="2D Brownian Motion Path",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        showlegend=True,
        width=600,
        height=600
    )
    
    return fig

def create_price_plot(simulation_data: Dict[str, List], initial_price: float) -> go.Figure:
    """Create stock price plot for geometric Brownian motion."""
    fig = go.Figure()
    
    time = simulation_data['time']
    prices = simulation_data['prices']
    
    fig.add_trace(go.Scatter(
        x=time, y=prices,
        mode='lines',
        name='Stock Price',
        line=dict(color='blue', width=2),
        hovertemplate='Time: %{x:.2f}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Add initial price line
    fig.add_hline(y=initial_price, line_dash="dash", line_color="gray",
                  annotation_text=f"Initial Price: ${initial_price}")
    
    fig.update_layout(
        title="Geometric Brownian Motion - Stock Price",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        showlegend=True
    )
    
    return fig

def create_distance_analysis(time: List[float], x_positions: List[List[float]], 
                           y_positions: List[List[float]]) -> go.Figure:
    """Create distance vs time analysis plot."""
    fig = go.Figure()
    
    # Calculate distances over time for each particle
    for i, (x_pos, y_pos) in enumerate(zip(x_positions, y_positions)):
        distances = [np.sqrt(x**2 + y**2) for x, y in zip(x_pos, y_pos)]
        
        fig.add_trace(go.Scatter(
            x=time, y=distances,
            mode='lines',
            name=f'Particle {i+1}',
            line=dict(width=1),
            opacity=0.7
        ))
    
    # Add theoretical curve (√t)
    theoretical = [np.sqrt(2 * t) for t in time]  # Assuming volatility = 1
    fig.add_trace(go.Scatter(
        x=time, y=theoretical,
        mode='lines',
        name='Theoretical (√t)',
        line=dict(color='red', width=3, dash='dash')
    ))
    
    fig.update_layout(
        title="Distance from Origin vs Time",
        xaxis_title="Time",
        yaxis_title="Distance from Origin",
        showlegend=True
    )
    
    return fig

def main():
    st.title("🌊 Brownian Motion Simulator")
    st.markdown("Explore random walks and stochastic processes")
    
    # Sidebar controls
    st.sidebar.header("Simulation Settings")
    
    simulation_type = st.sidebar.selectbox(
        "Simulation Type:",
        ["2D Brownian Motion", "Stock Price (GBM)"]
    )
    
    n_steps = st.sidebar.slider(
        "Number of Steps:",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )
    
    dt = st.sidebar.slider(
        "Time Step (dt):",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01
    )
    
    if simulation_type == "2D Brownian Motion":
        drift = st.sidebar.slider(
            "Drift (μ):",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1
        )
        
        volatility = st.sidebar.slider(
            "Volatility (σ):",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1
        )
        
        n_particles = st.sidebar.slider(
            "Number of Particles:",
            min_value=1,
            max_value=10,
            value=1,
            step=1
        )
    else:
        initial_price = st.sidebar.slider(
            "Initial Price ($):",
            min_value=10.0,
            max_value=1000.0,
            value=100.0,
            step=10.0
        )
        
        drift = st.sidebar.slider(
            "Annual Drift (μ):",
            min_value=-0.5,
            max_value=0.5,
            value=0.05,
            step=0.01
        )
        
        volatility = st.sidebar.slider(
            "Annual Volatility (σ):",
            min_value=0.1,
            max_value=1.0,
            value=0.2,
            step=0.01
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Simulation")
        
        if st.button("Run Simulation", type="primary"):
            if simulation_type == "2D Brownian Motion":
                # Run Brownian motion simulation
                simulation_data = simulate_brownian_motion(n_steps, dt, drift, volatility, n_particles)
                
                # Create path plot
                fig = create_path_plot(simulation_data, n_particles > 1)
                st.plotly_chart(fig, use_container_width=True)
                
                # Distance analysis
                if n_particles > 1:
                    dist_fig = create_distance_analysis(
                        simulation_data['time'], 
                        simulation_data['x_positions'], 
                        simulation_data['y_positions']
                    )
                    st.plotly_chart(dist_fig, use_container_width=True)
                
            else:
                # Run geometric Brownian motion simulation
                simulation_data = simulate_geometric_brownian_motion(
                    n_steps, dt, initial_price, drift, volatility
                )
                
                # Create price plot
                fig = create_price_plot(simulation_data, initial_price)
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Statistics")
        
        if 'simulation_data' in locals():
            if simulation_type == "2D Brownian Motion":
                # Calculate statistics
                stats = calculate_statistics(
                    [simulation_data['x_positions'], simulation_data['y_positions']], 
                    simulation_data['time']
                )
                
                st.metric("Mean Distance", f"{stats['mean_distance']:.3f}")
                st.metric("Theoretical Distance", f"{stats['theoretical_distance']:.3f}")
                st.metric("Std Distance", f"{stats['std_distance']:.3f}")
                
                # Final positions
                st.write("**Final Positions:**")
                for i in range(min(n_particles, 5)):  # Show first 5 particles
                    x_final = simulation_data['x_positions'][i][-1]
                    y_final = simulation_data['y_positions'][i][-1]
                    distance = np.sqrt(x_final**2 + y_final**2)
                    st.write(f"Particle {i+1}: ({x_final:.3f}, {y_final:.3f}) - Distance: {distance:.3f}")
                
            else:
                # Stock price statistics
                prices = simulation_data['prices']
                final_price = prices[-1]
                total_return = (final_price - initial_price) / initial_price
                
                st.metric("Final Price", f"${final_price:.2f}")
                st.metric("Total Return", f"{total_return:.1%}")
                st.metric("Max Price", f"${max(prices):.2f}")
                st.metric("Min Price", f"${min(prices):.2f}")
    
    # Theory section
    st.subheader("🧮 Theory Behind Brownian Motion")
    st.markdown("""
    **Mathematical Model:**
    - **Standard Brownian Motion:** dX = μdt + σdW
    - **Geometric Brownian Motion:** dS = S(μdt + σdW)
    
    **Key Properties:**
    1. **Continuous Paths:** No jumps or discontinuities
    2. **Independent Increments:** Future movement independent of past
    3. **Normal Distribution:** Increments follow normal distribution
    4. **Scaling:** Distance scales as √t (Einstein's result)
    
    **Applications:**
    - **Physics:** Particle diffusion, heat transfer
    - **Finance:** Stock prices, option pricing (Black-Scholes)
    - **Biology:** Molecular motion, population dynamics
    - **Chemistry:** Diffusion processes
    """)
    
    # Dynamic problem
    st.subheader("🧩 Dynamic Problem")
    st.markdown("""
    **Challenge:** Simulate a stock price with μ=0.1 and σ=0.2. What's the probability it doubles in 1 year?
    
    **Your Task:**
    1. Set up GBM simulation with given parameters
    2. Run 1000 simulations
    3. Count how many times the price doubles
    4. Compare with theoretical probability
    
    **Advanced Challenges:**
    - **Variant A:** What if the stock pays dividends?
    - **Variant B:** Model with mean reversion (Ornstein-Uhlenbeck process)
    - **Variant C:** Multi-asset portfolio simulation
    
    **Real-World Applications:**
    - Option pricing and risk management
    - Portfolio optimization
    - Interest rate modeling
    - Credit risk assessment
    """)
    
    # Learning objectives
    st.subheader("🎓 Learning Objectives")
    st.markdown("""
    **Key Insights:**
    - **Randomness in Nature:** Many natural processes are inherently random
    - **Mathematical Modeling:** Complex systems can be modeled with simple equations
    - **Time Evolution:** How systems change over time under uncertainty
    - **Risk and Uncertainty:** Quantifying and managing randomness
    
    **Mathematical Concepts:**
    - **Stochastic Differential Equations:** Modeling random processes
    - **Ito's Lemma:** Calculus for random functions
    - **Martingales:** Fair games and no-arbitrage
    - **Diffusion Processes:** Continuous-time random walks
    
    **Computational Skills:**
    - **Monte Carlo Simulation:** Numerical solution of SDEs
    - **Statistical Analysis:** Analyzing simulation results
    - **Visualization:** Understanding complex random processes
    - **Parameter Estimation:** Fitting models to data
    """)

if __name__ == "__main__":
    main()


