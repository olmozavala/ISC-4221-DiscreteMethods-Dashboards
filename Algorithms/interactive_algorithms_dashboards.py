import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import time
import random
from typing import List, Tuple
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Interactive Discrete Algorithms",
    page_icon="🧮",
    layout="wide"
)

# Navigation
st.sidebar.title("🧮 Algorithm Dashboards")
dashboard = st.sidebar.selectbox(
    "Choose a Dashboard:",
    [
        "Brute Force Sorting Visualizer",
        "Search Algorithm Comparison", 
        "Greedy Coin Change Simulator",
        "Big-O Complexity Explorer",
        "Algorithm Strategy Decision Tree"
    ]
)

# Utility functions
def generate_random_array(size: int, min_val: int = 1, max_val: int = 100) -> List[int]:
    """Generate a random array of given size."""
    return [random.randint(min_val, max_val) for _ in range(size)]

def selection_sort_steps(arr: List[int]) -> List[Tuple[List[int], str, int]]:
    """Perform selection sort and return each step."""
    steps = []
    arr_copy = arr.copy()
    n = len(arr_copy)
    
    for i in range(n - 1):
        min_idx = i
        for j in range(i + 1, n):
            if arr_copy[j] < arr_copy[min_idx]:
                min_idx = j
        
        if min_idx != i:
            arr_copy[i], arr_copy[min_idx] = arr_copy[min_idx], arr_copy[i]
            steps.append((arr_copy.copy(), f"Swapped {arr_copy[i]} and {arr_copy[min_idx]}", i))
        else:
            steps.append((arr_copy.copy(), f"No swap needed at position {i}", i))
    
    return steps

def bubble_sort_steps(arr: List[int]) -> List[Tuple[List[int], str, int]]:
    """Perform bubble sort and return each step."""
    steps = []
    arr_copy = arr.copy()
    n = len(arr_copy)
    
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr_copy[j] > arr_copy[j + 1]:
                arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
                swapped = True
                steps.append((arr_copy.copy(), f"Swapped {arr_copy[j+1]} and {arr_copy[j]}", j))
        
        if not swapped:
            break
    
    return steps

def sequential_search(arr: List[int], target: int) -> Tuple[int, List[int]]:
    """Perform sequential search and return steps."""
    steps = []
    for i, val in enumerate(arr):
        steps.append(i)
        if val == target:
            return i, steps
    return -1, steps

def binary_search(arr: List[int], target: int) -> Tuple[int, List[int]]:
    """Perform binary search and return steps."""
    steps = []
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        steps.append(mid)
        
        if arr[mid] == target:
            return mid, steps
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1, steps

def greedy_coin_change(amount: int, coins: List[int]) -> Tuple[List[int], int]:
    """Perform greedy coin change algorithm."""
    coins.sort(reverse=True)
    result = []
    remaining = amount
    
    for coin in coins:
        while remaining >= coin:
            result.append(coin)
            remaining -= coin
    
    return result, len(result)

def optimal_coin_change(amount: int, coins: List[int]) -> Tuple[List[int], int]:
    """Find optimal coin change using dynamic programming."""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    coin_used = [-1] * (amount + 1)
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                coin_used[i] = coin
    
    if dp[amount] == float('inf'):
        return [], 0
    
    # Reconstruct solution
    result = []
    current = amount
    while current > 0:
        coin = coin_used[current]
        result.append(coin)
        current -= coin
    
    return result, int(dp[amount])

# Dashboard 1: Brute Force Sorting Visualizer
if dashboard == "Brute Force Sorting Visualizer":
    st.title("🔍 Brute Force Sorting Visualizer")
    st.markdown("Explore how brute force algorithms work by visualizing Selection Sort and Bubble Sort step by step.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Controls
        array_size = st.slider("Array Size", 5, 20, 8)
        algorithm = st.selectbox("Algorithm", ["Selection Sort", "Bubble Sort"])
        
        if st.button("Generate Random Array"):
            st.session_state.array = generate_random_array(array_size)
        
        if st.button("Sort Step by Step"):
            if 'array' in st.session_state:
                if algorithm == "Selection Sort":
                    st.session_state.steps = selection_sort_steps(st.session_state.array)
                else:
                    st.session_state.steps = bubble_sort_steps(st.session_state.array)
                st.session_state.current_step = 0
    
    with col2:
        # Array display
        if 'array' in st.session_state:
            st.subheader("Current Array")
            
            # Create a bar chart with value labels
            fig = go.Figure(data=[
                go.Bar(
                    x=list(range(len(st.session_state.array))),
                    y=st.session_state.array,
                    marker_color='lightblue',
                    text=st.session_state.array,
                    textposition='outside',
                    textfont=dict(size=14, color='black')
                )
            ])
            fig.update_layout(
                title="Array Visualization",
                xaxis_title="Index",
                yaxis_title="Value",
                height=400,
                yaxis=dict(range=[0, max(st.session_state.array) + 10])
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Step-by-step execution
    if 'steps' in st.session_state and 'current_step' in st.session_state:
        st.subheader("Step-by-Step Execution")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("Previous Step") and st.session_state.current_step > 0:
                st.session_state.current_step -= 1
        
        with col2:
            st.write(f"Step {st.session_state.current_step + 1} of {len(st.session_state.steps)}")
            if st.session_state.current_step < len(st.session_state.steps):
                current_array, description, position = st.session_state.steps[st.session_state.current_step]
                st.write(f"**Action:** {description}")
                
                # Highlight the current position
                colors = ['lightblue'] * len(current_array)
                if position < len(colors):
                    colors[position] = 'red'
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(range(len(current_array))),
                        y=current_array,
                        marker_color=colors,
                        text=current_array,
                        textposition='outside',
                        textfont=dict(size=14, color='black')
                    )
                ])
                fig.update_layout(
                    title=f"Step {st.session_state.current_step + 1}",
                    xaxis_title="Index",
                    yaxis_title="Value",
                    height=400,
                    yaxis=dict(range=[0, max(current_array) + 10])
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            if st.button("Next Step") and st.session_state.current_step < len(st.session_state.steps) - 1:
                st.session_state.current_step += 1

# Dashboard 2: Search Algorithm Comparison
elif dashboard == "Search Algorithm Comparison":
    st.title("🔍 Search Algorithm Comparison")
    st.markdown("Compare Sequential Search (Brute Force) vs Binary Search (Decrease and Conquer)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        array_size = st.slider("Array Size", 10, 100, 20)
        target = st.number_input("Search Target", min_value=1, max_value=100, value=50)
        
        if st.button("Generate Sorted Array"):
            st.session_state.search_array = sorted(generate_random_array(array_size))
    
    with col2:
        if 'search_array' in st.session_state:
            st.subheader("Current Array")
            st.write(st.session_state.search_array)
            
            # Search comparison
            seq_result, seq_steps = sequential_search(st.session_state.search_array, target)
            bin_result, bin_steps = binary_search(st.session_state.search_array, target)
            
            st.subheader("Search Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Sequential Search (Brute Force)**")
                st.write(f"Result: {seq_result}")
                st.write(f"Steps: {len(seq_steps)}")
                st.write(f"Complexity: O(n)")
            
            with col2:
                st.write("**Binary Search (Decrease & Conquer)**")
                st.write(f"Result: {bin_result}")
                st.write(f"Steps: {len(bin_steps)}")
                st.write(f"Complexity: O(log n)")
            
            # Performance comparison chart
            sizes = list(range(10, 101, 10))
            seq_complexity = sizes
            bin_complexity = [np.log2(size) for size in sizes]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sizes, y=seq_complexity, name="Sequential Search O(n)", line=dict(color='red')))
            fig.add_trace(go.Scatter(x=sizes, y=bin_complexity, name="Binary Search O(log n)", line=dict(color='green')))
            fig.update_layout(
                title="Performance Comparison",
                xaxis_title="Array Size",
                yaxis_title="Number of Steps",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

# Dashboard 3: Greedy Coin Change Simulator
elif dashboard == "Greedy Coin Change Simulator":
    st.title("🪙 Greedy Coin Change Simulator")
    st.markdown("Explore when greedy algorithms work and when they fail")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        amount = st.slider("Amount to Make Change For", 1, 100, 41)
        
        coin_system = st.selectbox(
            "Coin System",
            ["US Coins (1, 5, 10, 25)", "Custom Coins", "Problematic Coins (1, 10, 25)"]
        )
        
        if coin_system == "US Coins (1, 5, 10, 25)":
            coins = [1, 5, 10, 25]
        elif coin_system == "Problematic Coins (1, 10, 25)":
            coins = [1, 10, 25]
        else:
            coins = [1, 5, 10, 20, 25]
        
        st.write(f"Available coins: {coins}")
    
    with col2:
        # Calculate solutions
        greedy_solution, greedy_count = greedy_coin_change(amount, coins)
        optimal_solution, optimal_count = optimal_coin_change(amount, coins)
        
        st.subheader("Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Greedy Solution**")
            st.write(f"Coins used: {greedy_solution}")
            st.write(f"Total coins: {greedy_count}")
        
        with col2:
            st.write("**Optimal Solution**")
            st.write(f"Coins used: {optimal_solution}")
            st.write(f"Total coins: {optimal_count}")
        
        # Comparison
        if greedy_count == optimal_count:
            st.success("✅ Greedy algorithm found the optimal solution!")
        else:
            st.error(f"❌ Greedy algorithm failed! Used {greedy_count} coins instead of {optimal_count}")
        
        # Visualization
        if greedy_solution:
            fig = go.Figure(data=[
                go.Bar(
                    x=[f"Greedy ({greedy_count})", f"Optimal ({optimal_count})"],
                    y=[greedy_count, optimal_count],
                    marker_color=['red', 'green']
                )
            ])
            fig.update_layout(
                title="Number of Coins Used",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

# Dashboard 4: Big-O Complexity Explorer
elif dashboard == "Big-O Complexity Explorer":
    st.title("📈 Big-O Complexity Explorer")
    st.markdown("Visualize different complexity classes and understand algorithm efficiency")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_functions = st.multiselect(
            "Select Functions to Compare:",
            ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n²)", "O(2ⁿ)"],
            default=["O(n)", "O(n²)"]
        )
        
        max_n = st.slider("Maximum Input Size", 10, 1000, 100)
    
    with col2:
        # Generate data
        n_values = list(range(1, max_n + 1))
        
        fig = go.Figure()
        
        if "O(1)" in selected_functions:
            fig.add_trace(go.Scatter(x=n_values, y=[1] * len(n_values), name="O(1)", line=dict(color='purple')))
        
        if "O(log n)" in selected_functions:
            fig.add_trace(go.Scatter(x=n_values, y=[np.log2(n) for n in n_values], name="O(log n)", line=dict(color='blue')))
        
        if "O(n)" in selected_functions:
            fig.add_trace(go.Scatter(x=n_values, y=n_values, name="O(n)", line=dict(color='green')))
        
        if "O(n log n)" in selected_functions:
            fig.add_trace(go.Scatter(x=n_values, y=[n * np.log2(n) for n in n_values], name="O(n log n)", line=dict(color='orange')))
        
        if "O(n²)" in selected_functions:
            fig.add_trace(go.Scatter(x=n_values, y=[n**2 for n in n_values], name="O(n²)", line=dict(color='red')))
        
        if "O(2ⁿ)" in selected_functions:
            fig.add_trace(go.Scatter(x=n_values, y=[2**n for n in n_values], name="O(2ⁿ)", line=dict(color='black')))
        
        fig.update_layout(
            title="Algorithm Complexity Comparison",
            xaxis_title="Input Size (n)",
            yaxis_title="Number of Operations",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance examples
    st.subheader("Real-World Performance Examples")
    
    example_sizes = [10, 100, 1000, 10000]
    data = []
    
    for size in example_sizes:
        row = {"Input Size": size}
        if "O(1)" in selected_functions:
            row["O(1)"] = 1
        if "O(log n)" in selected_functions:
            row["O(log n)"] = round(np.log2(size), 2)
        if "O(n)" in selected_functions:
            row["O(n)"] = size
        if "O(n log n)" in selected_functions:
            row["O(n log n)"] = round(size * np.log2(size), 2)
        if "O(n²)" in selected_functions:
            row["O(n²)"] = size**2
        if "O(2ⁿ)" in selected_functions:
            row["O(2ⁿ)"] = 2**size
        data.append(row)
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

# Dashboard 5: Algorithm Strategy Decision Tree
elif dashboard == "Algorithm Strategy Decision Tree":
    st.title("🌳 Algorithm Strategy Decision Tree")
    st.markdown("Learn to choose the right algorithmic strategy for different problems")
    
    problem_type = st.selectbox(
        "What type of problem are you solving?",
        ["Sorting", "Searching", "Optimization", "Path Finding", "Data Processing"]
    )
    
    if problem_type == "Sorting":
        st.subheader("Sorting Algorithm Selection")
        
        data_size = st.slider("Data Size", 10, 1000000, 1000)
        data_type = st.selectbox("Data Characteristics", ["Random", "Nearly Sorted", "Reversed", "Many Duplicates"])
        memory_constraint = st.checkbox("Memory Constraint")
        
        # Decision logic
        if data_size < 50:
            recommendation = "**Insertion Sort** - Simple and efficient for small datasets"
            reason = "For small datasets, simple algorithms often outperform complex ones due to lower overhead."
        elif data_size < 1000 and data_type == "Nearly Sorted":
            recommendation = "**Bubble Sort** - Good for nearly sorted data"
            reason = "Bubble sort can be very efficient when data is already nearly sorted."
        elif memory_constraint:
            recommendation = "**Selection Sort** - In-place sorting with minimal memory"
            reason = "Selection sort uses constant extra space regardless of input size."
        else:
            recommendation = "**Quick Sort or Merge Sort** - Efficient for large datasets"
            reason = "These divide-and-conquer algorithms provide O(n log n) average/worst-case performance."
        
        st.write("**Recommendation:**", recommendation)
        st.write("**Reason:**", reason)
    
    elif problem_type == "Searching":
        st.subheader("Search Algorithm Selection")
        
        data_size = st.slider("Data Size", 10, 1000000, 1000)
        data_sorted = st.selectbox("Data Status", ["Unsorted", "Sorted"])
        search_frequency = st.selectbox("Search Frequency", ["One-time", "Frequent"])
        
        if data_sorted == "Unsorted":
            if search_frequency == "One-time":
                recommendation = "**Sequential Search** - Simple and no preprocessing needed"
                reason = "For one-time searches on unsorted data, sequential search is straightforward."
            else:
                recommendation = "**Sort first, then Binary Search** - Transform and Conquer"
                reason = "The cost of sorting is amortized over multiple searches."
        else:
            recommendation = "**Binary Search** - Optimal for sorted data"
            reason = "Binary search provides O(log n) performance on sorted data."
        
        st.write("**Recommendation:**", recommendation)
        st.write("**Reason:**", reason)
    
    elif problem_type == "Optimization":
        st.subheader("Optimization Strategy Selection")
        
        problem_nature = st.selectbox("Problem Nature", ["Greedy Choice Works", "Greedy Choice Fails", "Unknown"])
        problem_size = st.slider("Problem Size", 10, 10000, 100)
        
        if problem_nature == "Greedy Choice Works":
            recommendation = "**Greedy Algorithm** - Fast and often optimal"
            reason = "When greedy choices lead to optimal solutions, these algorithms are very efficient."
        elif problem_nature == "Greedy Choice Fails":
            recommendation = "**Dynamic Programming or Branch and Bound** - Guaranteed optimal"
            reason = "When greedy fails, we need algorithms that explore multiple solution paths."
        else:
            recommendation = "**Try Greedy First, then Dynamic Programming** - Empirical approach"
            reason = "Start with greedy for speed, then use more sophisticated methods if needed."
        
        st.write("**Recommendation:**", recommendation)
        st.write("**Reason:**", reason)
    
    # Strategy comparison table
    st.subheader("Algorithm Strategy Comparison")
    
    strategies_data = {
        "Strategy": ["Brute Force", "Divide & Conquer", "Decrease & Conquer", "Transform & Conquer", "Greedy"],
        "Best For": ["Small problems", "Large problems", "Search problems", "Preprocessing helps", "Optimization"],
        "Complexity": ["Often O(n²) or worse", "Often O(n log n)", "Often O(log n)", "Varies", "Often O(n log n)"],
        "Guarantee": ["Always works", "Often optimal", "Often optimal", "Depends on transform", "May not be optimal"]
    }
    
    df = pd.DataFrame(strategies_data)
    st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Interactive Discrete Algorithms Dashboard** - Created for educational purposes")
st.markdown("Explore different algorithmic strategies and understand their trade-offs through hands-on experimentation.") 