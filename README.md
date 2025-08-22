# Interactive Algorithms Dashboards

A collection of interactive educational dashboards for learning discrete algorithms concepts, built with Streamlit and various visualization libraries.

## Available Modules

- **Graph Algorithms**: Interactive graph theory concepts and algorithms (see `Graphs/` directory)
- **Discrete Algorithms**: Fundamental algorithmic strategies and complexity analysis (see `README_Algorithms.md`)
- **Probability & Monte Carlo**: Statistical simulations and probability concepts (see `ProbabilityMonteCarlo/` directory)
- **Image Processing**: Computer vision and image manipulation algorithms (see `ImageProcessing/` directory)
- **Data Mining**: Machine learning and data analysis algorithms (see `DataMining/` directory)

## Features

- **Graph Representations**: Explore adjacency matrices, adjacency lists, and edge lists
- **Graph Traversal**: Step-by-step visualization of BFS and DFS algorithms
- **Shortest Path**: Interactive demonstration of Dijkstra's algorithm
- **Minimum Spanning Tree**: Kruskal's algorithm with Union-Find visualization

## Installation

This project uses `uv` for dependency management. Make sure you have `uv` installed:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Run the dashboard**:
   ```bash
   uv run streamlit run interactive_graph_dashboards.py
   ```

   Or use the launcher script:
   ```bash
   python run_dashboard.py
   ```

3. **Open your browser** and navigate to `http://localhost:8501`

## Dependencies

- Python >= 3.8.1
- streamlit >= 1.28.0
- networkx >= 3.0
- plotly >= 5.15.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0

## Testing

Run the test suite to verify everything is working:

```bash
uv run python test_dashboard.py
```

## Recent Fixes

The following issues have been resolved:

1. **Python version compatibility**: Updated `requires-python` to `>=3.8.1` to fix dependency conflicts
2. **Missing scipy dependency**: Added scipy to dependencies for NetworkX adjacency matrix functionality
3. **Edge color handling**: Fixed potential errors when `edge_colors` parameter is `None`
4. **Node color validation**: Added proper validation for node color arrays
5. **Session state management**: Improved session state parameter comparison logic

## Usage

1. **Choose a Dashboard**: Use the sidebar to select between different graph algorithm dashboards
2. **Interactive Elements**: Each dashboard includes interactive buttons and controls to explore the algorithms
3. **Step-by-Step Visualization**: Watch algorithms execute step by step with detailed explanations
4. **Dynamic Problems**: Try the challenge problems to test your understanding

## Dashboard Sections

### Graph Representations
- Compare different graph data structures
- Analyze space and time complexity trade-offs
- Interactive highlighting of vertex connections

### Graph Traversal
- Visualize BFS vs DFS algorithms
- Step through traversal processes
- Compare traversal orders for different starting points

### Shortest Path
- Watch Dijkstra's algorithm in action
- Compare with BFS for unweighted graphs
- Interactive path finding challenges

### Minimum Spanning Tree
- Kruskal's algorithm step-by-step
- Union-Find cycle detection
- Dynamic edge removal experiments

## Troubleshooting

If you encounter issues:

1. **Dependency errors**: Run `uv sync` to reinstall dependencies
2. **Port conflicts**: Change the port in the launch command: `--server.port 8502`
3. **Browser issues**: Try accessing `http://localhost:8501` directly

## Development

To contribute or modify the dashboard:

1. Install development dependencies: `uv sync --extra dev`
2. Run tests: `uv run python test_dashboard.py`
3. Format code: `uv run black interactive_graph_dashboards.py`
4. Lint code: `uv run flake8 interactive_graph_dashboards.py`

## License

This project is for educational purposes. 