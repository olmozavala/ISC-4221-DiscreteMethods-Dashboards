"""
Interactive Dashboard 1: Probability Building Blocks Explorer
Topic: Sample spaces, events, and random variables
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Any
import pandas as pd

def calculate_sample_space(experiment_type: str, num_items: int) -> Dict[str, Any]:
    """Calculate sample space for different experiment types."""
    if experiment_type == "Dice Roll":
        outcomes = list(range(1, 7))
        if num_items == 1:
            sample_space = outcomes
        else:
            # For multiple dice, we consider sums
            min_sum = num_items
            max_sum = 6 * num_items
            sample_space = list(range(min_sum, max_sum + 1))
            
    elif experiment_type == "Coin Flip":
        outcomes = ["H", "T"]
        if num_items == 1:
            sample_space = outcomes
        else:
            # For multiple coins, we consider number of heads
            sample_space = list(range(num_items + 1))
            
    elif experiment_type == "Card Draw":
        # Simplified: just consider card values 1-13
        sample_space = list(range(1, 14))
        
    return {
        "sample_space": sample_space,
        "outcomes": outcomes if experiment_type != "Card Draw" else list(range(1, 14))
    }

def simulate_experiment(experiment_type: str, num_items: int, num_trials: int) -> List[int]:
    """Simulate the experiment multiple times."""
    if experiment_type == "Dice Roll":
        if num_items == 1:
            return np.random.randint(1, 7, num_trials).tolist()
        else:
            return np.sum(np.random.randint(1, 7, (num_trials, num_items)), axis=1).tolist()
            
    elif experiment_type == "Coin Flip":
        if num_items == 1:
            return np.random.choice([0, 1], num_trials).tolist()  # 0=T, 1=H
        else:
            return np.sum(np.random.choice([0, 1], (num_trials, num_items)), axis=1).tolist()
            
    elif experiment_type == "Card Draw":
        return np.random.randint(1, 14, num_trials).tolist()

def calculate_event_probability(event_values: List[int], sample_space: List[int], num_trials: int) -> float:
    """Calculate probability of an event based on simulation."""
    if num_trials == 0:
        return 0.0
    
    # Simulate and count occurrences
    simulations = simulate_experiment(st.session_state.experiment_type, st.session_state.num_items, num_trials)
    event_occurrences = sum(1 for outcome in simulations if outcome in event_values)
    return event_occurrences / num_trials

def main():
    st.title("🎲 Probability Building Blocks Explorer")
    st.markdown("Explore sample spaces, events, and random variables through interactive simulation")
    
    # Initialize session state
    if 'experiment_type' not in st.session_state:
        st.session_state.experiment_type = "Dice Roll"
    if 'num_items' not in st.session_state:
        st.session_state.num_items = 1
    if 'trials' not in st.session_state:
        st.session_state.trials = 100
    
    # Sidebar controls
    st.sidebar.header("Experiment Settings")
    
    experiment_type = st.sidebar.selectbox(
        "Choose Experiment Type:",
        ["Dice Roll", "Coin Flip", "Card Draw"],
        key="experiment_type"
    )
    
    num_items = st.sidebar.slider(
        "Number of Items:",
        min_value=1,
        max_value=10,
        value=st.session_state.num_items,
        key="num_items"
    )
    
    num_trials = st.sidebar.slider(
        "Number of Trials:",
        min_value=10,
        max_value=10000,
        value=st.session_state.trials,
        step=10,
        key="trials"
    )
    
    # Calculate sample space
    space_info = calculate_sample_space(experiment_type, num_items)
    sample_space = space_info["sample_space"]
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Sample Space")
        st.write(f"**Experiment:** {num_items} {experiment_type.lower()}")
        st.write(f"**Sample Space:** {sample_space}")
        st.write(f"**Size:** {len(sample_space)} possible outcomes")
        
        # Event definition
        st.subheader("🎯 Define Your Event")
        if experiment_type == "Dice Roll":
            if num_items == 1:
                event_values = st.multiselect(
                    "Select outcomes for your event:",
                    options=sample_space,
                    default=[1, 2, 3]
                )
            else:
                event_values = st.multiselect(
                    "Select sum values for your event:",
                    options=sample_space,
                    default=sample_space[:3]
                )
        elif experiment_type == "Coin Flip":
            if num_items == 1:
                event_values = st.multiselect(
                    "Select outcomes (0=Tails, 1=Heads):",
                    options=[0, 1],
                    default=[1]
                )
            else:
                event_values = st.multiselect(
                    "Select number of heads:",
                    options=sample_space,
                    default=[0, 1]
                )
        else:  # Card Draw
            event_values = st.multiselect(
                "Select card values:",
                options=sample_space,
                default=[1, 2, 3]
            )
    
    with col2:
        st.subheader("📈 Simulation Results")
        
        if st.button("Run Simulation", type="primary"):
            # Run simulation
            results = simulate_experiment(experiment_type, num_items, num_trials)
            
            # Calculate empirical probability
            event_count = sum(1 for outcome in results if outcome in event_values)
            empirical_prob = event_count / num_trials
            
            # Theoretical probability (simplified)
            theoretical_prob = len(event_values) / len(sample_space)
            
            # Display results
            st.metric("Event Occurrences", event_count)
            st.metric("Empirical Probability", f"{empirical_prob:.4f}")
            st.metric("Theoretical Probability", f"{theoretical_prob:.4f}")
            
            # Create frequency distribution
            fig = go.Figure()
            
            # Count frequencies
            unique, counts = np.unique(results, return_counts=True)
            freq_dict = dict(zip(unique, counts))
            
            # Fill in missing values
            all_freqs = [freq_dict.get(x, 0) for x in sample_space]
            
            # Create bar chart
            fig.add_trace(go.Bar(
                x=sample_space,
                y=all_freqs,
                name="Frequency",
                marker_color='lightblue'
            ))
            
            # Highlight event values
            event_freqs = [freq_dict.get(x, 0) for x in event_values]
            fig.add_trace(go.Bar(
                x=event_values,
                y=event_freqs,
                name="Event Outcomes",
                marker_color='red'
            ))
            
            fig.update_layout(
                title="Frequency Distribution",
                xaxis_title="Outcome",
                yaxis_title="Frequency",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Learning objectives
    st.subheader("🎓 Learning Objectives")
    st.markdown("""
    **Key Concepts to Explore:**
    - **Random Variable:** The numerical outcome of your experiment
    - **Sample Space:** All possible outcomes (Ω)
    - **Event:** A subset of the sample space you're interested in
    - **Probability:** The likelihood of your event occurring
    
    **Try This Challenge:**
    Design an experiment with 3 dice where the sum is between 10-15. 
    What's the probability of this event? Verify by running simulations!
    """)
    
    # Dynamic problem section
    st.subheader("🧩 Dynamic Problem")
    st.markdown("""
    **Problem:** You're designing a game with 2 dice. You win if the sum is:
    - **Easy:** 7 or 11
    - **Medium:** Between 8 and 10 (inclusive)
    - **Hard:** Exactly 12
    
    **Your Task:**
    1. Set up the experiment (2 dice)
    2. Define each event
    3. Run simulations to estimate probabilities
    4. Which event is most likely? Least likely?
    5. Calculate theoretical probabilities and compare!
    """)

if __name__ == "__main__":
    main()


