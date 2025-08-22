"""
Interactive Graph Algorithms Dashboards for Discrete Algorithms Education

This module provides interactive dashboards for teaching graph theory concepts
including graph representations, traversal algorithms, shortest path algorithms,
and minimum spanning trees.

Author: Educational Dashboard Creator
Date: 2025
"""

import warnings
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Any
import heapq
from collections import deque, defaultdict
import time

# Suppress Streamlit ScriptRunContext warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")


class GraphVisualizer:
    """Base class for graph visualization utilities."""
    
    def __init__(self) -> None:
        """Initialize the graph visualizer."""
        self.colors = {
            'default': '#1f77b4',
            'visited': '#ff7f0e',
            'current': '#d62728',
            'path': '#2ca02c',
            'mst': '#9467bd',
            'highlight': '#e377c2'
        }
    
    def create_network_plot(self, G: nx.Graph, 
                           pos: Optional[Dict] = None,
                           node_colors: Optional[List[str]] = None,
                           edge_colors: Optional[List[str]] = None,
                           title: str = "Graph Visualization") -> go.Figure:
        """
        Create an interactive network plot using Plotly.
        
        Args:
            G: NetworkX graph object
            pos: Node positions dictionary
            node_colors: List of colors for nodes
            edge_colors: List of colors for edges
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        if pos is None:
            pos = nx.spring_layout(G, seed=42)
        
        # Use single color for all edges
        edge_color = edge_colors[0] if edge_colors is not None and len(edge_colors) > 0 else self.colors['default']
        
        # Check if graph is directed
        is_directed = G.is_directed()
        
        if is_directed:
            # For directed graphs, create arrows using annotations
            edge_trace = go.Scatter(
                x=[], y=[],
                line=dict(width=2, color=edge_color),
                hoverinfo='none',
                mode='lines'
            )
            
            # Add edges as lines
            edge_x = []
            edge_y = []
            arrow_annotations = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # Add arrow annotation
                # Calculate arrow position (80% along the edge)
                arrow_x = x0 + 0.8 * (x1 - x0)
                arrow_y = y0 + 0.8 * (y1 - y0)
                
                # Calculate arrow direction
                dx = x1 - x0
                dy = y1 - y0
                length = (dx**2 + dy**2)**0.5
                if length > 0:
                    dx, dy = dx/length, dy/length
                
                # Create arrow annotation
                arrow_annotations.append(
                    dict(
                        x=arrow_x,
                        y=arrow_y,
                        xref="x", yref="y",
                        axref="x", ayref="y",
                        text="",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=2,
                        arrowcolor=edge_color,
                        ax=arrow_x - 0.12 * dx,
                        ay=arrow_y - 0.12 * dy
                    )
                )
                
                # Add weight annotation if edge has weight
                if 'weight' in G[edge[0]][edge[1]]:
                    weight_x = x0 + 0.5 * (x1 - x0)
                    weight_y = y0 + 0.5 * (y1 - y0)
                    arrow_annotations.append(
                        dict(
                            x=weight_x,
                            y=weight_y,
                            xref="x", yref="y",
                            text=str(G[edge[0]][edge[1]]['weight']),
                            showarrow=False,
                            font=dict(size=12, color="red", weight="bold"),
                            bgcolor="white",
                            bordercolor="red",
                            borderwidth=1
                        )
                    )
            
            edge_trace.x = edge_x
            edge_trace.y = edge_y
            
        else:
            # For undirected graphs, use simple lines
            edge_x = []
            edge_y = []
            arrow_annotations = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # Add weight annotation if edge has weight
                if 'weight' in G[edge[0]][edge[1]]:
                    weight_x = x0 + 0.5 * (x1 - x0)
                    weight_y = y0 + 0.5 * (y1 - y0)
                    arrow_annotations.append(
                        dict(
                            x=weight_x,
                            y=weight_y,
                            xref="x", yref="y",
                            text=str(G[edge[0]][edge[1]]['weight']),
                            showarrow=False,
                            font=dict(size=12, color="red", weight="bold"),
                            bgcolor="white",
                            bordercolor="red",
                            borderwidth=1
                        )
                    )
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color=edge_color),
                hoverinfo='none',
                mode='lines'
            )
        
        # Prepare node traces
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        if node_colors is None:
            node_colors = [self.colors['default']] * len(G.nodes())
        elif len(node_colors) != len(G.nodes()):
            # Pad or truncate to match number of nodes
            if len(node_colors) < len(G.nodes()):
                node_colors.extend([self.colors['default']] * (len(G.nodes()) - len(node_colors)))
            else:
                node_colors = node_colors[:len(G.nodes())]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[str(node) for node in G.nodes()],
            textposition="middle center",
            textfont=dict(color='white', size=16),
            marker=dict(
                size=50,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           height=320,  # Reduced height by ~20% from 400px
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           annotations=arrow_annotations
                       ))
        
        return fig


class GraphRepresentationDashboard:
    """Interactive dashboard for exploring graph representations."""
    
    def __init__(self) -> None:
        """Initialize the graph representation dashboard."""
        self.visualizer = GraphVisualizer()
        self.sample_graphs = {
            "Simple Undirected": self._create_simple_undirected(),
            "Directed Graph": self._create_directed_graph(),
            "Weighted Graph": self._create_weighted_graph(),
            "Social Network": self._create_social_network()
        }
    
    def _create_simple_undirected(self) -> nx.Graph:
        """Create a simple undirected graph."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (4, 5)])
        return G
    
    def _create_directed_graph(self) -> nx.DiGraph:
        """Create a directed graph."""
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 3), (3, 4), (4, 1)])
        return G
    
    def _create_weighted_graph(self) -> nx.Graph:
        """Create a weighted graph."""
        G = nx.Graph()
        edges = [(0, 1, 4), (0, 2, 2), (1, 2, 1), (1, 3, 5), (2, 3, 8), (2, 4, 10), (3, 4, 2)]
        G.add_weighted_edges_from(edges)
        return G
    
    def _create_social_network(self) -> nx.Graph:
        """Create a social network-like graph."""
        G = nx.Graph()
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 4), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)]
        G.add_edges_from(edges)
        return G
    
    def adjacency_matrix_to_dataframe(self, G: nx.Graph) -> pd.DataFrame:
        """Convert graph to adjacency matrix DataFrame."""
        nodes = sorted(G.nodes())
        matrix = nx.adjacency_matrix(G, nodelist=nodes).todense()
        return pd.DataFrame(matrix, index=nodes, columns=nodes)
    
    def adjacency_list_to_dict(self, G: nx.Graph) -> Dict:
        """Convert graph to adjacency list dictionary."""
        return {node: list(G.neighbors(node)) for node in sorted(G.nodes())}
    
    def edge_list_to_dataframe(self, G: nx.Graph) -> pd.DataFrame:
        """Convert graph to edge list DataFrame."""
        edges = list(G.edges())
        # Check if graph has weighted edges by looking at the first edge
        if edges and 'weight' in G[edges[0][0]][edges[0][1]]:
            weights = [G[u][v]['weight'] for u, v in edges]
            return pd.DataFrame({'From': [u for u, v in edges], 
                               'To': [v for u, v in edges], 
                               'Weight': weights})
        else:
            return pd.DataFrame({'From': [u for u, v in edges], 
                               'To': [v for u, v in edges]})
    
    def run(self) -> None:
        """Run the graph representation dashboard."""
        st.header("📊 Graph Representations Interactive Dashboard")
        st.markdown("""
        **Learning Goal**: Understand how graphs can be represented in different data structures
        and the trade-offs between memory usage and operation efficiency.
        """)
        
        # Graph selection
        graph_name = st.selectbox(
            "Choose a graph to explore:",
            list(self.sample_graphs.keys())
        )
        
        G = self.sample_graphs[graph_name]
        
        # Display graph visualization
        st.subheader("Graph Visualization")
        fig = self.visualizer.create_network_plot(G, title=f"{graph_name} Graph")
        st.plotly_chart(fig, use_container_width=True)
        
        # Representation tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Adjacency Matrix", "Adjacency List", "Edge List", "Analysis"])
        
        with tab1:
            st.subheader("Adjacency Matrix")
            st.markdown("""
            **What it shows**: A matrix where entry (i,j) is 1 if there's an edge between vertices i and j.
            **Space complexity**: O(V²) where V is the number of vertices.
            **Time to check edge**: O(1)
            """)
            
            adj_matrix = self.adjacency_matrix_to_dataframe(G)
            st.dataframe(adj_matrix, use_container_width=True)
            
            # Interactive element
            # Initialize session state
            if 'show_highlight' not in st.session_state:
                st.session_state.show_highlight = False
            if 'selected_vertex' not in st.session_state:
                st.session_state.selected_vertex = 0
            
            # Vertex selection dropdown
            vertex = st.selectbox("Select vertex:", sorted(G.nodes()), key="highlight_vertex")
            
            # Update session state when vertex changes
            if vertex != st.session_state.selected_vertex:
                st.session_state.selected_vertex = vertex
                st.session_state.show_highlight = False
            
            # Button to show/hide highlighting
            if st.button("Highlight vertex connections", key="highlight_button"):
                st.session_state.show_highlight = not st.session_state.show_highlight
            
            # Show highlighted matrix if button was clicked
            if st.session_state.show_highlight:
                neighbors = list(G.neighbors(vertex))
                
                # Create highlighted matrix
                highlighted_matrix = adj_matrix.copy().astype(str)  # Convert to string first
                for neighbor in neighbors:
                    highlighted_matrix.loc[vertex, neighbor] = f"**{highlighted_matrix.loc[vertex, neighbor]}**"
                
                st.markdown("**Highlighted connections for vertex {}:**".format(vertex))
                st.dataframe(highlighted_matrix, use_container_width=True)
        
        with tab2:
            st.subheader("Adjacency List")
            st.markdown("""
            **What it shows**: For each vertex, a list of its neighbors.
            **Space complexity**: O(V + E) where E is the number of edges.
            **Time to find neighbors**: O(degree(v))
            """)
            
            adj_list = self.adjacency_list_to_dict(G)
            for vertex, neighbors in adj_list.items():
                st.write(f"**Vertex {vertex}**: {neighbors}")
            
            # Interactive element
            # Initialize session state for degree analysis
            if 'show_degree' not in st.session_state:
                st.session_state.show_degree = False
            if 'degree_vertex' not in st.session_state:
                st.session_state.degree_vertex = 0
            
            # Vertex selection dropdown
            vertex = st.selectbox("Select vertex to analyze:", sorted(G.nodes()), key="degree_vertex_select")
            
            # Update session state when vertex changes
            if vertex != st.session_state.degree_vertex:
                st.session_state.degree_vertex = vertex
                st.session_state.show_degree = False
            
            # Button to show/hide degree analysis
            if st.button("Find vertex degree", key="degree_button"):
                st.session_state.show_degree = not st.session_state.show_degree
            
            # Show degree analysis if button was clicked
            if st.session_state.show_degree:
                degree = G.degree(vertex)
                st.success(f"Degree of vertex {vertex}: {degree}")
                
                if G.is_directed():
                    in_degree = G.in_degree(vertex)
                    out_degree = G.out_degree(vertex)
                    st.info(f"In-degree: {in_degree}, Out-degree: {out_degree}")
        
        with tab3:
            st.subheader("Edge List")
            st.markdown("""
            **What it shows**: Simple list of all edges in the graph.
            **Space complexity**: O(E) where E is the number of edges.
            **Time to check edge**: O(E)
            """)
            
            edge_df = self.edge_list_to_dataframe(G)
            st.dataframe(edge_df, use_container_width=True)
            
            # Interactive element
            # Initialize session state for edge counting
            if 'show_edge_count' not in st.session_state:
                st.session_state.show_edge_count = False
            
            # Button to show/hide edge count
            if st.button("Count edges", key="count_edges_button"):
                st.session_state.show_edge_count = not st.session_state.show_edge_count
            
            # Show edge count if button was clicked
            if st.session_state.show_edge_count:
                edge_count = len(G.edges())
                vertex_count = len(G.nodes())
                st.info(f"Graph has {vertex_count} vertices and {edge_count} edges")
                
                # Check if graph has weighted edges
                edges = list(G.edges())
                if edges and 'weight' in G[edges[0][0]][edges[0][1]]:
                    total_weight = sum(G[u][v]['weight'] for u, v in G.edges())
                    st.info(f"Total edge weight: {total_weight}")
        
        with tab4:
            st.subheader("Representation Analysis")
            
            # Calculate metrics
            V = len(G.nodes())
            E = len(G.edges())
            
            metrics = {
                "Vertices": str(V),
                "Edges": str(E),
                "Adjacency Matrix Space": f"O(V²) = O({V}²) = O({V*V})",
                "Adjacency List Space": f"O(V+E) = O({V}+{E}) = O({V+E})",
                "Edge List Space": f"O(E) = O({E})"
            }
            
            # Create DataFrame with explicit string dtype to avoid Arrow conversion issues
            metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
            metrics_df = metrics_df.astype(str)
            st.dataframe(metrics_df)
            
            # Interactive comparison
            st.subheader("When to use each representation?")
            
            use_case = st.selectbox(
                "Select a use case:",
                ["Sparse graph (few edges)", "Dense graph (many edges)", 
                 "Frequent edge lookups", "Frequent neighbor traversals"]
            )
            
            if use_case == "Sparse graph (few edges)":
                st.success("**Recommendation**: Adjacency List - Most space efficient for sparse graphs")
            elif use_case == "Dense graph (many edges)":
                st.success("**Recommendation**: Adjacency Matrix - Fast edge lookups, space overhead acceptable")
            elif use_case == "Frequent edge lookups":
                st.success("**Recommendation**: Adjacency Matrix - O(1) edge existence checks")
            elif use_case == "Frequent neighbor traversals":
                st.success("**Recommendation**: Adjacency List - O(degree(v)) neighbor access")


class GraphTraversalDashboard:
    """Interactive dashboard for exploring graph traversal algorithms."""
    
    def __init__(self) -> None:
        """Initialize the graph traversal dashboard."""
        self.visualizer = GraphVisualizer()
        self.sample_graphs = {
            "Simple Tree": self._create_simple_tree(),
            "Complex Graph": self._create_complex_graph(),
            "Grid Graph": self._create_grid_graph()
        }
    
    def _create_simple_tree(self) -> nx.Graph:
        """Create a simple tree structure."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
        return G
    
    def _create_complex_graph(self) -> nx.Graph:
        """Create a more complex graph with cycles."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)])
        return G
    
    def _create_grid_graph(self) -> nx.Graph:
        """Create a grid-like graph."""
        G = nx.grid_2d_graph(3, 3)
        # Convert to simple node labels
        mapping = {(i, j): i * 3 + j for i in range(3) for j in range(3)}
        G = nx.relabel_nodes(G, mapping)
        return G
    
    def bfs_traversal(self, G: nx.Graph, start: int) -> List[Tuple[int, List[int], List[int]]]:
        """
        Perform BFS traversal and return step-by-step results.
        
        Args:
            G: NetworkX graph
            start: Starting vertex
            
        Returns:
            List of (current_vertex, queue_state, visited_vertices) tuples
        """
        visited = set()
        queue = deque([start])
        steps = []
        
        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                steps.append((current, list(queue), list(visited)))
                
                for neighbor in G.neighbors(current):
                    if neighbor not in visited and neighbor not in queue:
                        queue.append(neighbor)
        
        return steps
    
    def dfs_traversal(self, G: nx.Graph, start: int) -> List[Tuple[int, List[int], List[int]]]:
        """
        Perform DFS traversal and return step-by-step results.
        
        Args:
            G: NetworkX graph
            start: Starting vertex
            
        Returns:
            List of (current_vertex, stack_state, visited_vertices) tuples
        """
        visited = set()
        stack = [start]
        steps = []
        
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                steps.append((current, list(stack), list(visited)))
                
                for neighbor in reversed(list(G.neighbors(current))):
                    if neighbor not in visited and neighbor not in stack:
                        stack.append(neighbor)
        
        return steps
    
    def run(self) -> None:
        """Run the graph traversal dashboard."""
        st.header("🔍 Graph Traversal Algorithms Interactive Dashboard")
        st.markdown("""
        **Learning Goal**: Understand the difference between Depth-First Search (DFS) and 
        Breadth-First Search (BFS) and when to use each algorithm.
        """)
        
        # Graph selection
        graph_name = st.selectbox(
            "Choose a graph to traverse:",
            list(self.sample_graphs.keys())
        )
        
        G = self.sample_graphs[graph_name]
        
        # Display graph visualization
        st.subheader("Graph Visualization")
        fig = self.visualizer.create_network_plot(G, title=f"{graph_name} Graph")
        st.plotly_chart(fig, use_container_width=True)
        
        # Algorithm selection
        algorithm = st.selectbox(
            "Choose traversal algorithm:",
            ["Breadth-First Search (BFS)", "Depth-First Search (DFS)"]
        )
        
        # Starting vertex selection
        start_vertex = st.selectbox(
            "Select starting vertex:",
            sorted(G.nodes())
        )
        
        # Interactive traversal
        # Initialize session state for traversal
        if 'show_traversal' not in st.session_state:
            st.session_state.show_traversal = False
        if 'show_comparison' not in st.session_state:
            st.session_state.show_comparison = False
        if 'traversal_steps' not in st.session_state:
            st.session_state.traversal_steps = None
        if 'traversal_algorithm' not in st.session_state:
            st.session_state.traversal_algorithm = None
        
        # Check if parameters changed
        current_params = (algorithm, start_vertex, graph_name)
        if ('traversal_params' not in st.session_state or 
            st.session_state.traversal_params != current_params):
            st.session_state.traversal_params = current_params
            st.session_state.show_traversal = False
            st.session_state.show_comparison = False
            st.session_state.traversal_steps = None
        
        # Start traversal button
        if st.button("Start Traversal", key="start_traversal_button"):
            st.session_state.show_traversal = True
            st.session_state.show_comparison = False
            
            if algorithm == "Breadth-First Search (BFS)":
                steps = self.bfs_traversal(G, start_vertex)
                st.session_state.traversal_steps = steps
                st.session_state.traversal_algorithm = algorithm
                st.subheader("BFS Traversal Steps")
                st.markdown("""
                **BFS explores level by level, like ripples in a pond.**
                - Uses a **Queue** (FIFO: First In, First Out)
                - Guarantees shortest path in unweighted graphs
                - Good for: finding shortest paths, network broadcasting
                """)
            else:
                steps = self.dfs_traversal(G, start_vertex)
                st.session_state.traversal_steps = steps
                st.session_state.traversal_algorithm = algorithm
                st.subheader("DFS Traversal Steps")
                st.markdown("""
                **DFS goes deep down one path before backtracking.**
                - Uses a **Stack** (LIFO: Last In, First Out)
                - Finds a path (not necessarily shortest)
                - Good for: cycle detection, topological sorting, maze solving
                """)
            
            # Display steps
            for i, (current, data_structure, visited) in enumerate(steps):
                with st.expander(f"Step {i+1}: Visit vertex {current}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Current Vertex:**")
                        st.success(current)
                    
                    with col2:
                        st.write("**Data Structure State:**")
                        if algorithm == "Breadth-First Search (BFS)":
                            st.write(f"Queue: {data_structure}")
                        else:
                            st.write(f"Stack: {data_structure}")
                    
                    with col3:
                        st.write("**Visited Vertices:**")
                        st.write(visited)
            
            # Final result
            st.subheader("Traversal Complete!")
            traversal_order = [step[0] for step in steps]
            st.success(f"Traversal order: {' → '.join(map(str, traversal_order))}")
        
        # Show traversal results if available
        if st.session_state.show_traversal and st.session_state.traversal_steps:
            steps = st.session_state.traversal_steps
            algorithm = st.session_state.traversal_algorithm
            
            if algorithm == "Breadth-First Search (BFS)":
                st.subheader("BFS Traversal Steps")
                st.markdown("""
                **BFS explores level by level, like ripples in a pond.**
                - Uses a **Queue** (FIFO: First In, First Out)
                - Guarantees shortest path in unweighted graphs
                - Good for: finding shortest paths, network broadcasting
                """)
            else:
                st.subheader("DFS Traversal Steps")
                st.markdown("""
                **DFS goes deep down one path before backtracking.**
                - Uses a **Stack** (LIFO: Last In, First Out)
                - Finds a path (not necessarily shortest)
                - Good for: cycle detection, topological sorting, maze solving
                """)
            
            # Display steps
            for i, (current, data_structure, visited) in enumerate(steps):
                with st.expander(f"Step {i+1}: Visit vertex {current}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Current Vertex:**")
                        st.success(current)
                    
                    with col2:
                        st.write("**Data Structure State:**")
                        if algorithm == "Breadth-First Search (BFS)":
                            st.write(f"Queue: {data_structure}")
                        else:
                            st.write(f"Stack: {data_structure}")
                    
                    with col3:
                        st.write("**Visited Vertices:**")
                        st.write(visited)
            
            # Final result
            st.subheader("Traversal Complete!")
            traversal_order = [step[0] for step in steps]
            st.success(f"Traversal order: {' → '.join(map(str, traversal_order))}")
            
            # Interactive comparison button
            if st.button("Compare with other algorithm", key="compare_algorithm_button"):
                st.session_state.show_comparison = True
            
            # Show comparison if requested
            if st.session_state.show_comparison:
                other_algorithm = "Depth-First Search (DFS)" if algorithm == "Breadth-First Search (BFS)" else "Breadth-First Search (BFS)"
                if other_algorithm == "Breadth-First Search (BFS)":
                    other_steps = self.bfs_traversal(G, start_vertex)
                else:
                    other_steps = self.dfs_traversal(G, start_vertex)
                
                other_order = [step[0] for step in other_steps]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{algorithm}:**")
                    st.write(' → '.join(map(str, traversal_order)))
                
                with col2:
                    st.write(f"**{other_algorithm}:**")
                    st.write(' → '.join(map(str, other_order)))
        
        # Dynamic problem
        st.subheader("🎯 Dynamic Problem: Path Finding Challenge")
        st.markdown("""
        **Problem**: Find the shortest path between two vertices using BFS.
        **Challenge**: Adjust the graph and observe how BFS always finds the shortest path!
        """)
        
        target_vertex = st.selectbox(
            "Select target vertex:",
            sorted(G.nodes()),
            key="target_vertex"
        )
        
        if st.button("Find shortest path"):
            # Simple BFS path finding
            queue = deque([(start_vertex, [start_vertex])])
            visited = set()
            
            while queue:
                current, path = queue.popleft()
                if current == target_vertex:
                    st.success(f"Shortest path from {start_vertex} to {target_vertex}: {' → '.join(map(str, path))}")
                    st.info(f"Path length: {len(path) - 1} edges")
                    break
                
                if current not in visited:
                    visited.add(current)
                    for neighbor in G.neighbors(current):
                        if neighbor not in visited:
                            queue.append((neighbor, path + [neighbor]))
            else:
                st.error(f"No path found from {start_vertex} to {target_vertex}")


class ShortestPathDashboard:
    """Interactive dashboard for exploring shortest path algorithms."""
    
    def __init__(self) -> None:
        """Initialize the shortest path dashboard."""
        self.visualizer = GraphVisualizer()
        self.sample_graphs = {
            "Simple Weighted": self._create_simple_weighted(),
            "Complex Network": self._create_complex_network(),
            "Grid with Weights": self._create_weighted_grid()
        }
    
    def _create_simple_weighted(self) -> nx.Graph:
        """Create a simple weighted graph."""
        G = nx.Graph()
        edges = [(0, 1, 4), (0, 2, 2), (1, 2, 1), (1, 3, 5), (2, 3, 8), (2, 4, 10), (3, 4, 2)]
        G.add_weighted_edges_from(edges)
        return G
    
    def _create_complex_network(self) -> nx.Graph:
        """Create a more complex weighted network."""
        G = nx.Graph()
        edges = [(0, 1, 3), (0, 2, 5), (1, 2, 2), (1, 3, 4), (2, 3, 1), (2, 4, 6), (3, 4, 2), (3, 5, 7), (4, 5, 3)]
        G.add_weighted_edges_from(edges)
        return G
    
    def _create_weighted_grid(self) -> nx.Graph:
        """Create a weighted grid graph."""
        G = nx.grid_2d_graph(3, 3)
        # Add random weights
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.randint(1, 10)
        
        # Convert to simple node labels
        mapping = {(i, j): i * 3 + j for i in range(3) for j in range(3)}
        G = nx.relabel_nodes(G, mapping)
        return G
    
    def dijkstra_step_by_step(self, G: nx.Graph, start: int) -> List[Dict]:
        """
        Perform Dijkstra's algorithm step by step.
        
        Args:
            G: NetworkX weighted graph
            start: Starting vertex
            
        Returns:
            List of step dictionaries with distances and current vertex
        """
        distances = {node: float('infinity') for node in G.nodes()}
        distances[start] = 0
        visited = set()
        steps = []
        
        while len(visited) < len(G.nodes()):
            # Find unvisited node with minimum distance
            current = min(
                (node for node in G.nodes() if node not in visited),
                key=lambda x: distances[x]
            )
            
            if distances[current] == float('infinity'):
                break  # No more reachable nodes
            
            visited.add(current)
            steps.append({
                'current': current,
                'distances': distances.copy(),
                'visited': visited.copy()
            })
            
            # Relax edges
            for neighbor in G.neighbors(current):
                if neighbor not in visited:
                    new_distance = distances[current] + G[current][neighbor]['weight']
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
        
        return steps
    
    def run(self) -> None:
        """Run the shortest path dashboard."""
        st.header("🗺️ Shortest Path Algorithms Interactive Dashboard")
        st.markdown("""
        **Learning Goal**: Understand Dijkstra's algorithm and how it finds the shortest path
        in weighted graphs using a greedy approach with a priority queue.
        """)
        
        # Graph selection
        graph_name = st.selectbox(
            "Choose a weighted graph:",
            list(self.sample_graphs.keys())
        )
        
        G = self.sample_graphs[graph_name]
        
        # Display graph visualization with weights
        st.subheader("Weighted Graph Visualization")
        
        # Create edge labels for weights
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=2, color=self.visualizer.colors['default']),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers+text',
            hoverinfo='text',
            text=[],
            textposition="middle center",
            textfont=dict(color='white', size=16),
            marker=dict(size=50, color=self.visualizer.colors['default'])
        )
        
        pos = nx.spring_layout(G, seed=42)
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_annotations = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Add weight annotation
            edge_annotations.append(
                dict(
                    x=(x0 + x1) / 2,
                    y=(y0 + y1) / 2,
                    xref="x", yref="y",
                    text=str(G[edge[0]][edge[1]]['weight']),
                    showarrow=False,
                    font=dict(size=14, color="red", weight="bold"),
                    bgcolor="white",
                    bordercolor="red",
                    borderwidth=1
                )
            )
        
        edge_trace.x = edge_x
        edge_trace.y = edge_y
        
        # Add nodes
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_trace.x = node_x
        node_trace.y = node_y
        node_trace.text = [str(node) for node in G.nodes()]
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f"{graph_name} Graph (Edge weights shown in red)",
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           annotations=edge_annotations
                       ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Algorithm parameters
        start_vertex = st.selectbox(
            "Select starting vertex:",
            sorted(G.nodes())
        )
        
        if st.button("Run Dijkstra's Algorithm"):
            st.subheader("Dijkstra's Algorithm Step-by-Step")
            st.markdown("""
            **How it works**:
            1. Start with distance 0 to source, ∞ to all others
            2. Always pick the unvisited node with minimum distance
            3. Relax all edges from current node
            4. Repeat until all nodes visited
            """)
            
            steps = self.dijkstra_step_by_step(G, start_vertex)
            
            for i, step in enumerate(steps):
                with st.expander(f"Step {i+1}: Visit vertex {step['current']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Current Vertex:**")
                        st.success(step['current'])
                        
                        st.write("**Visited Vertices:**")
                        st.write(sorted(step['visited']))
                    
                    with col2:
                        st.write("**Distance Array:**")
                        distance_df = pd.DataFrame(
                            list(step['distances'].items()),
                            columns=['Vertex', 'Distance']
                        )
                        st.dataframe(distance_df, use_container_width=True)
            
            # Final result
            final_distances = steps[-1]['distances']
            st.subheader("Final Shortest Path Distances")
            
            result_df = pd.DataFrame(
                list(final_distances.items()),
                columns=['Vertex', 'Shortest Distance']
            )
            st.dataframe(result_df, use_container_width=True)
            
            # Path reconstruction
            st.subheader("Path Reconstruction")
            target = st.selectbox(
                "Select target vertex to find path:",
                sorted(G.nodes()),
                key="dijkstra_target"
            )
            
            if st.button("Find path to target"):
                # Simple path reconstruction (in practice, you'd store predecessors)
                if final_distances[target] == float('infinity'):
                    st.error(f"No path exists from {start_vertex} to {target}")
                else:
                    st.success(f"Shortest distance from {start_vertex} to {target}: {final_distances[target]}")
        
        # Interactive comparison with BFS
        st.subheader("🎯 Dynamic Problem: BFS vs Dijkstra's")
        st.markdown("""
        **Challenge**: Compare how BFS and Dijkstra's handle the same graph.
        BFS finds shortest path in terms of number of edges, while Dijkstra's finds shortest path in terms of total weight.
        """)
        
        if st.button("Compare BFS and Dijkstra's"):
            # BFS shortest path (unweighted)
            bfs_distances = {}
            queue = deque([(start_vertex, 0)])
            visited = set()
            
            while queue:
                current, distance = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    bfs_distances[current] = distance
                    
                    for neighbor in G.neighbors(current):
                        if neighbor not in visited:
                            queue.append((neighbor, distance + 1))
            
            # Compare results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**BFS Results (edge count):**")
                bfs_df = pd.DataFrame(
                    list(bfs_distances.items()),
                    columns=['Vertex', 'Edge Count']
                )
                st.dataframe(bfs_df, use_container_width=True)
            
            with col2:
                st.write("**Dijkstra's Results (total weight):**")
                dijkstra_df = pd.DataFrame(
                    list(final_distances.items()),
                    columns=['Vertex', 'Total Weight']
                )
                st.dataframe(dijkstra_df, use_container_width=True)
            
            # Show difference
            st.subheader("Key Insight")
            st.info("""
            **BFS** finds the path with the fewest edges (ignoring weights).
            **Dijkstra's** finds the path with the lowest total weight.
            For unweighted graphs, they give the same result!
            """)


class MSTDashboard:
    """Interactive dashboard for exploring Minimum Spanning Tree algorithms."""
    
    def __init__(self) -> None:
        """Initialize the MST dashboard."""
        self.visualizer = GraphVisualizer()
        self.sample_graphs = {
            "Simple MST": self._create_simple_mst(),
            "Complex Network": self._create_complex_mst_network(),
            "Random Graph": self._create_random_mst_graph()
        }
    
    def _create_simple_mst(self) -> nx.Graph:
        """Create a simple graph for MST demonstration."""
        G = nx.Graph()
        edges = [(0, 1, 4), (0, 2, 2), (1, 2, 1), (1, 3, 5), (2, 3, 8), (2, 4, 10), (3, 4, 2)]
        G.add_weighted_edges_from(edges)
        return G
    
    def _create_complex_mst_network(self) -> nx.Graph:
        """Create a more complex network for MST."""
        G = nx.Graph()
        edges = [(0, 1, 3), (0, 2, 5), (1, 2, 2), (1, 3, 4), (2, 3, 1), (2, 4, 6), (3, 4, 2), (3, 5, 7), (4, 5, 3), (0, 3, 8)]
        G.add_weighted_edges_from(edges)
        return G
    
    def _create_random_mst_graph(self) -> nx.Graph:
        """Create a random graph for MST."""
        G = nx.erdos_renyi_graph(8, 0.4, seed=42)
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.randint(1, 10)
        return G
    
    def kruskal_step_by_step(self, G: nx.Graph) -> List[Dict]:
        """
        Perform Kruskal's algorithm step by step.
        
        Args:
            G: NetworkX weighted graph
            
        Returns:
            List of step dictionaries with edge consideration and MST state
        """
        edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])
        mst_edges = []
        components = {node: {node} for node in G.nodes()}
        steps = []
        
        for edge in edges:
            u, v, weight = edge
            
            # Check if adding this edge creates a cycle
            if components[u] != components[v]:
                # No cycle, add edge to MST
                mst_edges.append((u, v, weight))
                
                # Merge components
                merged_component = components[u].union(components[v])
                for node in merged_component:
                    components[node] = merged_component
                
                steps.append({
                    'edge': (u, v, weight),
                    'action': 'added',
                    'mst_edges': mst_edges.copy(),
                    'components': {k: v.copy() for k, v in components.items()}
                })
            else:
                steps.append({
                    'edge': (u, v, weight),
                    'action': 'rejected',
                    'mst_edges': mst_edges.copy(),
                    'components': {k: v.copy() for k, v in components.items()}
                })
            
            # Stop if we have V-1 edges
            if len(mst_edges) == len(G.nodes()) - 1:
                break
        
        return steps
    
    def run(self) -> None:
        """Run the MST dashboard."""
        st.header("🌳 Minimum Spanning Tree Interactive Dashboard")
        st.markdown("""
        **Learning Goal**: Understand Kruskal's algorithm and how it finds the minimum spanning tree
        by sorting edges and using Union-Find to detect cycles.
        """)
        
        # Graph selection
        graph_name = st.selectbox(
            "Choose a graph for MST:",
            list(self.sample_graphs.keys())
        )
        
        G = self.sample_graphs[graph_name]
        
        # Display graph visualization
        st.subheader("Original Graph")
        fig = self.visualizer.create_network_plot(G, title=f"{graph_name} Graph")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show edge weights
        st.subheader("Edge Weights")
        edge_data = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
        edge_df = pd.DataFrame(edge_data, columns=['From', 'To', 'Weight'])
        edge_df = edge_df.sort_values('Weight')
        st.dataframe(edge_df, use_container_width=True)
        
        if st.button("Run Kruskal's Algorithm"):
            st.subheader("Kruskal's Algorithm Step-by-Step")
            st.markdown("""
            **How it works**:
            1. Sort all edges by weight (ascending)
            2. Consider each edge in order
            3. Add edge if it doesn't create a cycle
            4. Use Union-Find to track connected components
            """)
            
            steps = self.kruskal_step_by_step(G)
            
            for i, step in enumerate(steps):
                u, v, weight = step['edge']
                action = step['action']
                
                with st.expander(f"Step {i+1}: Consider edge ({u}, {v}) with weight {weight} - {action.upper()}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Edge Considered:**")
                        if action == 'added':
                            st.success(f"({u}, {v}) - Weight: {weight} - ADDED")
                        else:
                            st.error(f"({u}, {v}) - Weight: {weight} - REJECTED (would create cycle)")
                    
                    with col2:
                        st.write("**Current MST Edges:**")
                        if step['mst_edges']:
                            mst_df = pd.DataFrame(
                                step['mst_edges'],
                                columns=['From', 'To', 'Weight']
                            )
                            st.dataframe(mst_df, use_container_width=True)
                        else:
                            st.write("No edges added yet")
                    
                    # Show connected components
                    st.write("**Connected Components:**")
                    components = step['components']
                    unique_components = set()
                    for comp in components.values():
                        unique_components.add(tuple(sorted(comp)))
                    
                    for comp in unique_components:
                        st.write(f"Component: {list(comp)}")
            
            # Final result
            final_mst_edges = steps[-1]['mst_edges']
            total_weight = sum(weight for _, _, weight in final_mst_edges)
            
            st.subheader("Final Minimum Spanning Tree")
            st.success(f"Total MST weight: {total_weight}")
            
            # Create MST visualization
            MST = nx.Graph()
            MST.add_weighted_edges_from(final_mst_edges)
            
            fig_mst = self.visualizer.create_network_plot(
                MST, 
                title="Minimum Spanning Tree",
                edge_colors=[self.visualizer.colors['mst']]
            )
            st.plotly_chart(fig_mst, use_container_width=True)
            
            # Show MST edges
            mst_df = pd.DataFrame(
                final_mst_edges,
                columns=['From', 'To', 'Weight']
            )
            st.dataframe(mst_df, use_container_width=True)
        
        # Interactive problem
        st.subheader("🎯 Dynamic Problem: MST Challenge")
        st.markdown("""
        **Challenge**: What happens if you remove an edge from the original graph?
        How does it affect the MST?
        """)
        
        edge_to_remove = st.selectbox(
            "Select edge to remove:",
            [(u, v) for u, v in G.edges()],
            format_func=lambda x: f"({x[0]}, {x[1]}) - Weight: {G[x[0]][x[1]]['weight']}"
        )
        
        if st.button("Remove edge and recalculate MST"):
            # Create new graph without the selected edge
            G_new = G.copy()
            G_new.remove_edge(*edge_to_remove)
            
            if nx.is_connected(G_new):
                # Recalculate MST
                new_steps = self.kruskal_step_by_step(G_new)
                new_mst_edges = new_steps[-1]['mst_edges']
                new_total_weight = sum(weight for _, _, weight in new_mst_edges)
                
                st.success(f"New MST weight: {new_total_weight}")
                st.info(f"Weight difference: {new_total_weight - total_weight}")
                
                # Show new MST
                MST_new = nx.Graph()
                MST_new.add_weighted_edges_from(new_mst_edges)
                
                fig_new_mst = self.visualizer.create_network_plot(
                    MST_new,
                    title="New Minimum Spanning Tree",
                    edge_colors=[self.visualizer.colors['mst']]
                )
                st.plotly_chart(fig_new_mst, use_container_width=True)
            else:
                st.error("Removing this edge disconnects the graph! No MST exists.")


def main() -> None:
    """Main function to run the interactive dashboards."""
    st.set_page_config(
        page_title="Interactive Graph Algorithms",
        page_icon="📊",
        layout="wide"
    )
    
    # Custom CSS to reduce header space and set max-width to 1200px
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px !important;
    }
    .main header {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    h1 {
        font-size: 1.8rem !important;
        margin-bottom: 0.1rem !important;
        margin-top: 0.1rem !important;
        line-height: 1.2 !important;
    }
    h2 {
        font-size: 1.4rem !important;
        margin-bottom: 0.1rem !important;
        margin-top: 0.1rem !important;
        line-height: 1.2 !important;
    }
    h3 {
        font-size: 1.2rem !important;
        margin-bottom: 0.1rem !important;
        margin-top: 0.1rem !important;
        line-height: 1.2 !important;
    }
    /* Reduce spacing between paragraphs and increase font size */
    p {
        margin-bottom: 0.1rem !important;
        margin-top: 0.1rem !important;
        line-height: 1.3 !important;
        font-size: 1.3rem !important;
    }
    /* Increase font size for other text elements */
    label, span, div {
        font-size: 1.3rem !important;
    }
    /* Increase font size for selectbox and other form elements */
    .stSelectbox, .stButton, .stTextInput {
        font-size: 1.3rem !important;
    }
    /* Increase font size specifically for tables */
    .stDataFrame, .stDataFrameGlideDataEditor, .gdg-wmyidgi {
        font-size: 1.5rem !important;
    }
    /* Target table cells specifically */
    .stDataFrame td, .stDataFrame th, .gdg-wmyidgi td, .gdg-wmyidgi th {
        font-size: 1.5rem !important;
    }
    /* Target the specific Streamlit container classes */
    .st-emotion-cache-zy6yx3 {
        max-width: 80% !important;
        padding: 1rem 1rem 3rem !important;
    }
    .st-emotion-cache-4rsbii {
        align-items: flex-start !important;
    }
    /* Media query for wider screens */
    @media (min-width: calc(736px + 8rem)) {
        .st-emotion-cache-zy6yx3 {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    /* Additional targeting for any other container classes */
    [data-testid="stAppViewContainer"] > div {
        max-width: 80% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 Interactive Graph Algorithms Learning Dashboard")
    st.markdown("""
    Welcome to the interactive graph algorithms tutorial! This dashboard provides hands-on 
    exploration of fundamental graph theory concepts and algorithms.
    """)
    
    # Navigation
    dashboard_type = st.sidebar.selectbox(
        "Choose a Dashboard:",
        ["Graph Representations", "Graph Traversal", "Shortest Path", "Minimum Spanning Tree"]
    )
    
    if dashboard_type == "Graph Representations":
        dashboard = GraphRepresentationDashboard()
        dashboard.run()
    elif dashboard_type == "Graph Traversal":
        dashboard = GraphTraversalDashboard()
        dashboard.run()
    elif dashboard_type == "Shortest Path":
        dashboard = ShortestPathDashboard()
        dashboard.run()
    elif dashboard_type == "Minimum Spanning Tree":
        dashboard = MSTDashboard()
        dashboard.run()
    
    # Summary table
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Learning Summary")
    
    summary_data = {
        "Graph Representations": {
            "Dashboard": "Adjacency Matrix/List/Edge List",
            "Student Action": "Compare representations, analyze trade-offs",
            "Key Insight": "Different representations optimize for different operations"
        },
        "Graph Traversal": {
            "Dashboard": "BFS vs DFS visualization",
            "Student Action": "Step through algorithms, compare traversal orders",
            "Key Insight": "BFS finds shortest paths, DFS explores deeply"
        },
        "Shortest Path": {
            "Dashboard": "Dijkstra's algorithm step-by-step",
            "Student Action": "Watch greedy choices, compare with BFS",
            "Key Insight": "Greedy approach works for non-negative weights"
        },
        "Minimum Spanning Tree": {
            "Dashboard": "Kruskal's algorithm with Union-Find",
            "Student Action": "Add edges in order, observe cycle detection",
            "Key Insight": "Greedy edge selection builds optimal tree"
        }
    }
    
    for topic, info in summary_data.items():
        with st.sidebar.expander(topic):
            st.write(f"**Dashboard:** {info['Dashboard']}")
            st.write(f"**Student Action:** {info['Student Action']}")
            st.write(f"**Key Insight:** {info['Key Insight']}")


if __name__ == "__main__":
    main() 