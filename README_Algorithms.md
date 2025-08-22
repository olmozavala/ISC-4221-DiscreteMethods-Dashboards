# Interactive Discrete Algorithms Dashboard

This interactive dashboard provides hands-on exploration of fundamental algorithmic concepts covered in the Discrete Algorithms tutorial. Students can experiment with different algorithms, observe their behavior, and understand the trade-offs between different approaches.

## Features

### 🔍 Brute Force Sorting Visualizer
- **Topic Covered:** Brute Force Strategy - Selection Sort and Bubble Sort
- **Interactive Elements:** 
  - Array size slider (5-20 elements)
  - Algorithm selector (Selection Sort vs Bubble Sort)
  - Step-by-step execution with visual feedback
  - Real-time array visualization with color coding
- **Learning Goal:** Understand the "generate and test" philosophy and observe O(n²) complexity
- **Dynamic Problem:** Predict steps for different array sizes and verify with the visualizer

### 🔍 Search Algorithm Comparison
- **Topic Covered:** Sequential Search vs Binary Search (Brute Force vs Decrease and Conquer)
- **Interactive Elements:**
  - Array size selector (10-1000 elements)
  - Search target input
  - Real-time performance comparison
  - Step counter and complexity display
- **Learning Goal:** Discover the dramatic efficiency difference between O(n) and O(log n) algorithms
- **Dynamic Problem:** Find the "break-even point" where Binary Search becomes faster than Sequential Search

### 🪙 Greedy Coin Change Simulator
- **Topic Covered:** Greedy Algorithms and When They Fail
- **Interactive Elements:**
  - Amount slider (1-100)
  - Coin system selector (US coins, custom coins, problematic coins)
  - Step-by-step greedy choice visualization
  - Optimal vs greedy solution comparison
- **Learning Goal:** Understand when greedy algorithms work and when they fail
- **Dynamic Problem:** Design coin systems where greedy fails and explain why

### 📈 Big-O Complexity Explorer
- **Topic Covered:** Algorithm Efficiency and Big-O Notation
- **Interactive Elements:**
  - Function selector (O(1), O(log n), O(n), O(n log n), O(n²), O(2ⁿ))
  - Input size slider (1-1000)
  - Real-time complexity comparison graphs
  - Performance prediction calculator
- **Learning Goal:** Develop intuition for different complexity classes
- **Dynamic Problem:** Predict which algorithm will be fastest for given input sizes

### 🌳 Algorithm Strategy Decision Tree
- **Topic Covered:** Algorithm Design Strategy Selection
- **Interactive Elements:**
  - Problem type selector (sorting, searching, optimization, etc.)
  - Problem characteristics input
  - Strategy recommendation engine
  - Performance prediction
- **Learning Goal:** Learn to match algorithmic strategies to problem characteristics
- **Dynamic Problem:** Choose the best strategy for novel problem scenarios

## Installation and Setup

1. **Install Dependencies:**
   ```bash
   # Using pip
   pip install -e .
   
   # Or using uv (recommended)
   uv sync
   ```

2. **Run the Dashboard:**
   ```bash
   cd Algorithms
   streamlit run interactive_algorithms_dashboards.py
   ```

3. **Open in Browser:**
   The dashboard will automatically open in your default web browser at `http://localhost:8501`

## Usage Instructions

### For Students:
1. Start with the **Brute Force Sorting Visualizer** to understand basic algorithmic concepts
2. Move to **Search Algorithm Comparison** to see efficiency differences
3. Explore **Greedy Coin Change Simulator** to understand when greedy approaches work
4. Use **Big-O Complexity Explorer** to develop intuition for algorithm performance
5. Practice with **Algorithm Strategy Decision Tree** to learn problem-solving approaches

### For Instructors:
1. **Introduction:** Use the sorting visualizer to demonstrate brute force concepts
2. **Comparison:** Use search comparison to show efficiency trade-offs
3. **Critical Thinking:** Use greedy simulator to discuss when algorithms fail
4. **Analysis:** Use Big-O explorer to teach complexity analysis
5. **Application:** Use decision tree to help students choose appropriate strategies

## Educational Benefits

- **Visual Learning:** Students can see algorithms in action with real-time visualizations
- **Hands-on Experimentation:** Interactive controls allow students to test hypotheses
- **Immediate Feedback:** Real-time results help students understand concepts quickly
- **Comparative Analysis:** Side-by-side comparisons highlight key differences
- **Problem-Solving Practice:** Decision tree helps develop algorithmic thinking

## Technical Details

- **Framework:** Streamlit for web interface
- **Visualization:** Plotly for interactive charts
- **Data Processing:** NumPy and Pandas for calculations
- **Language:** Python 3.8+
- **Dependencies:** Managed through pyproject.toml

## Contributing

This dashboard is designed for educational use. Feel free to:
- Add new algorithms
- Improve visualizations
- Add more interactive elements
- Create additional problem scenarios

## License

Educational use only. Created for the Discrete Algorithms course module. 