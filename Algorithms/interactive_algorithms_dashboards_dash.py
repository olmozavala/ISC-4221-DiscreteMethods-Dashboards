import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import random
from typing import List, Tuple, Dict, Any
import pandas as pd
from dash.exceptions import PreventUpdate

# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__, title="Interactive Discrete Algorithms", 
                external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

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

# App layout
app.layout = dbc.Container([
    # Hidden divs for storing state
    html.Div(id='current-array', style={'display': 'none'}),
    html.Div(id='sort-steps', style={'display': 'none'}),
    html.Div(id='current-step', style={'display': 'none'}),
    html.Div(id='search-array', style={'display': 'none'}),
    html.Div(id='current-search-array', style={'display': 'none'}),
    html.Div(id='dashboard-selector', style={'display': 'none'}, children='sorting'),
    
    # Main container with sidebar and content
    dbc.Row([
        # Sidebar
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("🧮 Algorithm Dashboards", className="mb-0")
                ]),
                dbc.CardBody([
                    dbc.Nav([
                        dbc.NavItem([
                            dbc.NavLink("🔍 Brute Force Sorting Visualizer", 
                                       id="nav-sorting", active=True, href="#")
                        ]),
                        dbc.NavItem([
                            dbc.NavLink("🔍 Search Algorithm Comparison", 
                                       id="nav-search", href="#")
                        ]),
                        dbc.NavItem([
                            dbc.NavLink("🪙 Greedy Coin Change Simulator", 
                                       id="nav-coin-change", href="#")
                        ]),
                        dbc.NavItem([
                            dbc.NavLink("📈 Big-O Complexity Explorer", 
                                       id="nav-complexity", href="#")
                        ]),
                        dbc.NavItem([
                            dbc.NavLink("🌳 Algorithm Strategy Decision Tree", 
                                       id="nav-strategy", href="#")
                        ])
                    ], vertical=True, pills=True)
                ])
            ], className="h-100")
        ], width=3),
        
        # Content area
        dbc.Col([
            html.Div(id='dashboard-content')
        ], width=9)
    ], className="mt-3")
], fluid=True)

# Callbacks for navigation
@app.callback(
    Output('dashboard-content', 'children'),
    [Input('nav-sorting', 'n_clicks'),
     Input('nav-search', 'n_clicks'),
     Input('nav-coin-change', 'n_clicks'),
     Input('nav-complexity', 'n_clicks'),
     Input('nav-strategy', 'n_clicks')]
)
def update_dashboard(sorting_clicks, search_clicks, coin_clicks, complexity_clicks, strategy_clicks):
    """Update the dashboard content based on navigation clicks."""
    
    ctx = callback_context
    if not ctx.triggered:
        return create_sorting_dashboard()  # Default to sorting
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'nav-sorting':
        return create_sorting_dashboard()
    elif button_id == 'nav-search':
        return create_search_dashboard()
    elif button_id == 'nav-coin-change':
        return create_coin_change_dashboard()
    elif button_id == 'nav-complexity':
        return create_complexity_dashboard()
    elif button_id == 'nav-strategy':
        return create_strategy_dashboard()
    else:
        return create_sorting_dashboard()  # Default to sorting

@app.callback(
    [Output('nav-sorting', 'active'),
     Output('nav-search', 'active'),
     Output('nav-coin-change', 'active'),
     Output('nav-complexity', 'active'),
     Output('nav-strategy', 'active')],
    [Input('nav-sorting', 'n_clicks'),
     Input('nav-search', 'n_clicks'),
     Input('nav-coin-change', 'n_clicks'),
     Input('nav-complexity', 'n_clicks'),
     Input('nav-strategy', 'n_clicks')]
)
def update_nav_active_states(sorting_clicks, search_clicks, coin_clicks, complexity_clicks, strategy_clicks):
    """Update the active states of navigation tabs."""
    
    ctx = callback_context
    if not ctx.triggered:
        # Default to sorting being active
        return True, False, False, False, False
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Set all to False first, then set the active one to True
    active_states = [False, False, False, False, False]
    
    if button_id == 'nav-sorting':
        active_states[0] = True
    elif button_id == 'nav-search':
        active_states[1] = True
    elif button_id == 'nav-coin-change':
        active_states[2] = True
    elif button_id == 'nav-complexity':
        active_states[3] = True
    elif button_id == 'nav-strategy':
        active_states[4] = True
    
    return active_states

def create_sorting_dashboard() -> dbc.Container:
    """Create the sorting visualizer dashboard."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🔍 Brute Force Sorting Visualizer", className="mb-3"),
                html.P("Explore how brute force algorithms work by visualizing Selection Sort and Bubble Sort step by step.", 
                       className="text-muted")
            ])
        ], className="mb-4"),
        
        # Controls and Array visualization in same row
        dbc.Row([
            # Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Controls"),
                    dbc.CardBody([
                        html.Div([
                            dbc.Label("Array Size:"),
                            dcc.Slider(
                                id='array-size', 
                                min=5, max=20, value=8, 
                                step=1,
                                marks={i: str(i) for i in range(5, 21, 5)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], className="mb-3"),
                        
                        html.Div([
                            dbc.Label("Algorithm:"),
                            dcc.Dropdown(
                                id='algorithm-selector',
                                options=[
                                    {'label': 'Selection Sort', 'value': 'selection'},
                                    {'label': 'Bubble Sort', 'value': 'bubble'}
                                ],
                                value='selection'
                            )
                        ], className="mb-3"),
                        
                        dbc.Button("Generate Random Array", id='generate-array-btn', 
                                 n_clicks=0, color="primary", className="mb-2 w-100"),
                        dbc.Button("Sort Step by Step", id='sort-btn', 
                                 n_clicks=0, color="success", className="w-100")
                    ])
                ])
            ], width=3),
            
            # Array visualization
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Current Array"),
                    dbc.CardBody([
                        dcc.Graph(id='array-graph', style={'height': '400px'})
                    ])
                ])
            ], width=9)
        ], className="mb-4"),
        
        # Step-by-step execution (below)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Step-by-Step Execution"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Previous Step", id='prev-step-btn', 
                                         n_clicks=0, color="secondary", className="w-100")
                            ], width=2),
                            dbc.Col([
                                html.Div(id='step-info', className="text-center p-2 bg-light rounded")
                            ], width=8),
                            dbc.Col([
                                dbc.Button("Next Step", id='next-step-btn', 
                                         n_clicks=0, color="secondary", className="w-100")
                            ], width=2)
                        ], className="mb-3"),
                        dcc.Graph(id='step-graph', style={'height': '400px'})
                    ])
                ], id='step-execution', style={'display': 'none'})
            ])
        ])
    ], fluid=True)

def create_search_dashboard() -> dbc.Container:
    """Create the search algorithm comparison dashboard."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🔍 Search Algorithm Comparison", className="mb-3"),
                html.P("Compare Sequential Search (Brute Force) vs Binary Search (Decrease and Conquer)", 
                       className="text-muted")
            ])
        ], className="mb-4"),
        
        dbc.Row([
            # Controls and Algorithm Code
            dbc.Col([
                # Controls
                dbc.Card([
                    dbc.CardHeader("Controls"),
                    dbc.CardBody([
                        html.Div([
                            dbc.Label("Array Size:"),
                            dcc.Slider(
                                id='search-array-size', 
                                min=10, max=100, value=20, 
                                step=1,
                                marks={i: str(i) for i in range(10, 101, 20)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], className="mb-3"),
                        
                        html.Div([
                            dbc.Label("Search Target:"),
                            dbc.Input(
                                id='search-target', 
                                type='number', 
                                value=50, 
                                min=1, 
                                max=100,
                                placeholder="Enter target value"
                            )
                        ], className="mb-3"),
                        
                        dbc.Button("Generate Sorted Array", id='generate-search-array-btn', 
                                 n_clicks=0, color="primary", className="mb-2 w-100"),
                        dbc.Button("Run Search", id='run-search-btn', 
                                 n_clicks=0, color="success", className="w-100")
                    ])
                ], className="mb-3"),
                
                # Algorithm Code Section
                dbc.Card([
                    dbc.CardHeader([
                        html.Span("Algorithm Code", className="me-2"),
                        dbc.Button("Show Code", id='toggle-code-btn', 
                                 n_clicks=0, size="sm", color="outline-secondary")
                    ], className="d-flex justify-content-between align-items-center"),
                    dbc.CardBody([
                        html.Div(id='algorithm-code-content', style={'display': 'none'})
                    ])
                ])
            ], width=3),
            
            # Array Display and Results
            dbc.Col([
                # Array Display
                dbc.Card([
                    dbc.CardHeader("Generated Sorted Array"),
                    dbc.CardBody([
                        html.Div(id='search-array-display', className="p-3 bg-light rounded"),
                        dcc.Graph(id='search-array-graph', style={'height': '300px'})
                    ])
                ], className="mb-3"),
                
                # Search Results
                dbc.Card([
                    dbc.CardHeader("Search Results"),
                    dbc.CardBody([
                        html.Div(id='search-results'),
                        dcc.Graph(id='search-comparison-graph', style={'height': '400px'})
                    ])
                ])
            ], width=9)
        ])
    ], fluid=True)

def create_coin_change_dashboard() -> dbc.Container:
    """Create the greedy coin change simulator dashboard."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🪙 Greedy Coin Change Simulator", className="mb-3"),
                html.P("Explore when greedy algorithms work and when they fail", 
                       className="text-muted")
            ])
        ], className="mb-4"),
        
        dbc.Row([
            # Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Controls"),
                    dbc.CardBody([
                        html.Div([
                            dbc.Label("Amount to Make Change For:"),
                            dcc.Slider(
                                id='change-amount', 
                                min=1, max=100, value=41, 
                                step=1,
                                marks={i: str(i) for i in range(0, 101, 20)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], className="mb-3"),
                        
                        html.Div([
                            dbc.Label("Coin System:"),
                            dcc.Dropdown(
                                id='coin-system',
                                options=[
                                    {'label': 'US Coins (1, 5, 10, 25)', 'value': 'us'},
                                    {'label': 'Custom Coins', 'value': 'custom'},
                                    {'label': 'Problematic Coins (1, 10, 25)', 'value': 'problematic'}
                                ],
                                value='us'
                            )
                        ], className="mb-3")
                    ])
                ])
            ], width=3),
            
            # Results
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Results"),
                    dbc.CardBody([
                        html.Div(id='coin-change-results'),
                        dcc.Graph(id='coin-change-graph', style={'height': '300px'})
                    ])
                ])
            ], width=9)
        ])
    ], fluid=True)

def create_complexity_dashboard() -> dbc.Container:
    """Create the Big-O complexity explorer dashboard."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("📈 Big-O Complexity Explorer", className="mb-3"),
                html.P("Visualize different complexity classes and understand algorithm efficiency", 
                       className="text-muted")
            ])
        ], className="mb-4"),
        
        dbc.Row([
            # Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Controls"),
                    dbc.CardBody([
                        html.Div([
                            dbc.Label("Select Functions to Compare:"),
                            dcc.Checklist(
                                id='complexity-functions',
                                options=[
                                    {'label': 'O(1)', 'value': 'O(1)'},
                                    {'label': 'O(log n)', 'value': 'O(log n)'},
                                    {'label': 'O(n)', 'value': 'O(n)'},
                                    {'label': 'O(n log n)', 'value': 'O(n log n)'},
                                    {'label': 'O(n²)', 'value': 'O(n²)'},
                                    {'label': 'O(2ⁿ)', 'value': 'O(2ⁿ)'}
                                ],
                                value=['O(n)', 'O(n²)']
                            )
                        ], className="mb-3"),
                        
                        html.Div([
                            dbc.Label("Maximum Input Size:"),
                            dcc.Slider(
                                id='max-n', 
                                min=10, max=1000, value=100, 
                                step=1,
                                marks={i: str(i) for i in range(0, 1001, 200)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], className="mb-3")
                    ])
                ])
            ], width=3),
            
            # Graph
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(id='complexity-graph', style={'height': '500px'})
                    ])
                ])
            ], width=9)
        ], className="mb-4"),
        
        # Performance table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Real-World Performance Examples"),
                    dbc.CardBody([
                        html.Div(id='performance-table')
                    ])
                ])
            ])
        ])
    ], fluid=True)

def create_strategy_dashboard() -> dbc.Container:
    """Create the algorithm strategy decision tree dashboard."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🌳 Algorithm Strategy Decision Tree", className="mb-3"),
                html.P("Learn to choose the right algorithmic strategy for different problems", 
                       className="text-muted")
            ])
        ], className="mb-4"),
        
        dbc.Row([
            # Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Problem Type"),
                    dbc.CardBody([
                        html.Div([
                            dcc.Dropdown(
                                id='problem-type',
                                options=[
                                    {'label': 'Sorting', 'value': 'sorting'},
                                    {'label': 'Searching', 'value': 'searching'},
                                    {'label': 'Optimization', 'value': 'optimization'}
                                ],
                                value='sorting'
                            )
                        ], className="mb-3"),
                        
                        html.Div(id='problem-specific-controls')
                    ])
                ])
            ], width=3),
            
            # Results
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Recommendation"),
                    dbc.CardBody([
                        html.Div(id='strategy-recommendation'),
                        html.Div(id='strategy-comparison-table')
                    ])
                ])
            ], width=9)
        ])
    ], fluid=True)

# Callbacks for sorting dashboard
@app.callback(
    [Output('current-array', 'children'),
     Output('array-graph', 'figure')],
    [Input('generate-array-btn', 'n_clicks')],
    [State('array-size', 'value')]
)
def generate_array(n_clicks: int, array_size: int) -> Tuple[str, go.Figure]:
    """Generate a random array and update the visualization."""
    if n_clicks == 0:
        raise PreventUpdate
    
    array = generate_random_array(array_size)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(range(len(array))),
            y=array,
            marker_color='lightblue',
            text=array,
            textposition='outside',
            textfont=dict(size=14, color='black')
        )
    ])
    fig.update_layout(
        title="Array Visualization",
        xaxis_title="Index",
        yaxis_title="Value",
        height=400,
        yaxis=dict(range=[0, max(array) + 10]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    return str(array), fig

@app.callback(
    [Output('sort-steps', 'children'),
     Output('step-execution', 'style')],
    [Input('sort-btn', 'n_clicks')],
    [State('current-array', 'children'),
     State('algorithm-selector', 'value')]
)
def start_sorting(n_clicks: int, array_str: str, algorithm: str) -> Tuple[str, Dict[str, str]]:
    """Start the sorting process and generate steps."""
    if n_clicks == 0 or not array_str:
        raise PreventUpdate
    
    array = eval(array_str)  # Convert string back to list
    
    if algorithm == 'selection':
        steps = selection_sort_steps(array)
    else:
        steps = bubble_sort_steps(array)
    
    return str(steps), {'display': 'block'}

@app.callback(
    [Output('step-info', 'children'),
     Output('step-graph', 'figure'),
     Output('current-step', 'children')],
    [Input('prev-step-btn', 'n_clicks'),
     Input('next-step-btn', 'n_clicks'),
     Input('sort-btn', 'n_clicks')],
    [State('sort-steps', 'children'),
     State('current-step', 'children'),
     State('current-array', 'children'),
     State('algorithm-selector', 'value')]
)
def update_step(prev_clicks: int, next_clicks: int, sort_clicks: int, steps_str: str, current_step_str: str, array_str: str, algorithm: str) -> Tuple[str, go.Figure, str]:
    """Update the current step in the sorting visualization."""
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # If sort button was clicked, generate initial step
    if button_id == 'sort-btn':
        if not array_str:
            raise PreventUpdate
        
        array = eval(array_str)  # Convert string back to list
        
        if algorithm == 'selection':
            steps = selection_sort_steps(array)
        else:
            steps = bubble_sort_steps(array)
        
        if steps:
            current_array, description, position = steps[0]
            
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
                title="Step 1",
                xaxis_title="Index",
                yaxis_title="Value",
                height=400,
                yaxis=dict(range=[0, max(current_array) + 10]),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12)
            )
            
            step_info = f"Step 1 of {len(steps)} - {description}"
            return step_info, fig, '0'
        else:
            fig = go.Figure()
            fig.update_layout(
                title="No steps available",
                xaxis_title="Index",
                yaxis_title="Value",
                height=400,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12)
            )
            return "No steps available", fig, '0'
    
    # Handle navigation buttons
    if not steps_str or not current_step_str:
        raise PreventUpdate
    
    steps = eval(steps_str)  # Convert string back to list
    current_step = int(current_step_str)
    
    if button_id == 'prev-step-btn' and current_step > 0:
        current_step -= 1
    elif button_id == 'next-step-btn' and current_step < len(steps) - 1:
        current_step += 1
    
    if current_step < len(steps):
        current_array, description, position = steps[current_step]
        
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
            title=f"Step {current_step + 1}",
            xaxis_title="Index",
            yaxis_title="Value",
            height=400,
            yaxis=dict(range=[0, max(current_array) + 10]),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        
        step_info = f"Step {current_step + 1} of {len(steps)} - {description}"
        return step_info, fig, str(current_step)
    else:
        # Handle edge case where step is out of bounds
        fig = go.Figure()
        fig.update_layout(
            title="Step out of bounds",
            xaxis_title="Index",
            yaxis_title="Value",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        return "Step out of bounds", fig, str(current_step)

# Callbacks for search dashboard
@app.callback(
    [Output('search-array', 'children'),
     Output('current-search-array', 'children'),
     Output('search-array-display', 'children'),
     Output('search-array-graph', 'figure'),
     Output('search-results', 'children'),
     Output('search-comparison-graph', 'figure')],
    [Input('generate-search-array-btn', 'n_clicks')],
    [State('search-array-size', 'value'),
     State('search-target', 'value')]
)
def generate_search_array_and_results(generate_clicks: int, array_size: int, target: int) -> Tuple[str, str, html.Div, go.Figure, html.Div, go.Figure]:
    """Generate a sorted array and perform search comparisons."""
    if generate_clicks == 0:
        raise PreventUpdate
    
    # Validate inputs
    if target is None:
        target = 50  # Default value
    if array_size is None:
        array_size = 20  # Default value
    
    # Generate sorted array
    array = sorted(generate_random_array(array_size))
    array_display_text = f"Sorted Array: {array}"
    
    # Perform searches
    seq_result, seq_steps = sequential_search(array, target)
    bin_result, bin_steps = binary_search(array, target)
    
    # Create array display
    array_display = html.Div([
        html.H5("Array Contents:", className="mb-2"),
        html.P(array_display_text, className="font-monospace"),
        html.Hr(),
        html.P(f"Array Size: {len(array)}", className="mb-1"),
        html.P(f"Search Target: {target}", className="mb-1"),
        html.P(f"Target Found: {'Yes' if target in array else 'No'}", className="mb-0")
    ])
    
    # Create array visualization
    array_fig = go.Figure()
    array_fig.add_trace(go.Bar(
        x=list(range(len(array))),
        y=array,
        text=array,
        textposition='outside',
        textfont=dict(size=12, color='black'),
        marker_color='lightblue',
        name='Array Values'
    ))
    
    # Highlight target if found
    if target in array:
        target_indices = [i for i, val in enumerate(array) if val == target]
        for idx in target_indices:
            array_fig.add_trace(go.Bar(
                x=[idx],
                y=[array[idx]],
                text=[array[idx]],
                textposition='outside',
                textfont=dict(size=12, color='white'),
                marker_color='red',
                name='Target Value',
                showlegend=False
            ))
    
    array_fig.update_layout(
        title="Array Visualization",
        xaxis_title="Index",
        yaxis_title="Value",
        height=300,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        showlegend=True
    )
    
    # Create results display
    results = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sequential Search (Brute Force)"),
                dbc.CardBody([
                    html.P(f"Result: {seq_result}"),
                    html.P(f"Steps: {len(seq_steps)}"),
                    html.P(f"Complexity: O(n)")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Binary Search (Decrease & Conquer)"),
                dbc.CardBody([
                    html.P(f"Result: {bin_result}"),
                    html.P(f"Steps: {len(bin_steps)}"),
                    html.P(f"Complexity: O(log n)")
                ])
            ])
        ], width=6)
    ])
    
    # Create comparison chart
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
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    return str(array), str(array), array_display, array_fig, results, fig

@app.callback(
    [Output('search-results', 'children', allow_duplicate=True),
     Output('search-comparison-graph', 'figure', allow_duplicate=True)],
    [Input('run-search-btn', 'n_clicks')],
    [State('current-search-array', 'children'),
     State('search-target', 'value')],
    prevent_initial_call=True
)
def run_search_on_existing_array(search_clicks: int, stored_array: str, target: int) -> Tuple[html.Div, go.Figure]:
    """Run search algorithms on the existing array with a new target."""
    if search_clicks == 0 or not stored_array:
        raise PreventUpdate
    
    # Validate target input
    if target is None:
        target = 50  # Default value
    
    # Parse the stored array
    try:
        array = eval(stored_array)  # Convert string back to list
    except:
        raise PreventUpdate
    
    # Perform searches
    seq_result, seq_steps = sequential_search(array, target)
    bin_result, bin_steps = binary_search(array, target)
    
    # Create results display
    results = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Sequential Search (Brute Force)"),
                dbc.CardBody([
                    html.P(f"Result: {seq_result}"),
                    html.P(f"Steps: {len(seq_steps)}"),
                    html.P(f"Complexity: O(n)")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Binary Search (Decrease & Conquer)"),
                dbc.CardBody([
                    html.P(f"Result: {bin_result}"),
                    html.P(f"Steps: {len(bin_steps)}"),
                    html.P(f"Complexity: O(log n)")
                ])
            ])
        ], width=6)
    ])
    
    # Create comparison chart
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
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    return results, fig

# Callback to update strategy recommendation based on problem-specific controls
@app.callback(
    Output('strategy-recommendation', 'children', allow_duplicate=True),
    [Input('problem-type', 'value')],
    [State('problem-specific-controls', 'children')],
    prevent_initial_call=True
)
def update_strategy_based_on_controls(problem_type: str, controls_children) -> html.Div:
    """Update strategy recommendation when problem-specific controls change."""
    
    # Helper function to extract values from controls HTML
    def extract_value_from_controls(controls_html, component_id):
        """Extract value from controls HTML by looking for component ID."""
        if not controls_html:
            return None
        
        # Convert to string if it's a component
        controls_str = str(controls_html)
        
        # Look for the component ID and extract its value
        if component_id in controls_str:
            # This is a simplified approach - we'll use default values for now
            return "default"
        return None
    
    if problem_type == 'sorting':
        # Extract values from sorting controls
        data_size = extract_value_from_controls(controls_children, 'sorting-data-size')
        data_type = extract_value_from_controls(controls_children, 'sorting-data-type')
        memory_constraint = extract_value_from_controls(controls_children, 'memory-constraint')
        
        # Provide specific recommendations based on typical values
        if data_size == "default":
            recommendation = "**Quick Sort or Merge Sort** - Recommended for typical datasets"
            reason = "For most sorting scenarios, Quick Sort (average O(n log n)) or Merge Sort (guaranteed O(n log n)) provide excellent performance. Quick Sort is often faster in practice, while Merge Sort guarantees consistent performance."
        else:
            recommendation = "**Quick Sort or Merge Sort** - Efficient for large datasets"
            reason = "These divide-and-conquer algorithms provide O(n log n) average/worst-case performance, making them ideal for most sorting tasks."
        
    elif problem_type == 'searching':
        # Extract values from searching controls
        data_status = extract_value_from_controls(controls_children, 'data-status')
        search_frequency = extract_value_from_controls(controls_children, 'search-frequency')
        
        if data_status == "sorted" or data_status == "default":
            recommendation = "**Binary Search** - Optimal for sorted data"
            reason = "Binary Search provides O(log n) performance on sorted data, making it the optimal choice for searching in sorted arrays."
        else:
            recommendation = "**Sequential Search** - Simple and effective for unsorted data"
            reason = "For unsorted data, Sequential Search is straightforward and doesn't require preprocessing. Consider sorting first if you'll search frequently."
        
    elif problem_type == 'optimization':
        # Extract values from optimization controls
        problem_nature = extract_value_from_controls(controls_children, 'problem-nature')
        
        if problem_nature == "greedy-works":
            recommendation = "**Greedy Algorithm** - Fast and often optimal"
            reason = "When greedy choices lead to optimal solutions, these algorithms are very efficient and straightforward to implement."
        elif problem_nature == "greedy-fails":
            recommendation = "**Dynamic Programming or Branch and Bound** - Guaranteed optimal"
            reason = "When greedy fails, we need algorithms that explore multiple solution paths to guarantee optimal solutions."
        else:
            recommendation = "**Try Greedy First, then Dynamic Programming** - Empirical approach"
            reason = "Start with greedy for speed, then use more sophisticated methods if needed. This approach balances efficiency with optimality."
        
    else:
        recommendation = "Select a problem type to get a recommendation."
        reason = ""
    
    return dbc.Card([
        dbc.CardHeader("Recommendation"),
        dbc.CardBody([
            html.H5("Recommendation:"),
            html.P(recommendation, className="font-weight-bold text-primary"),
            html.H5("Reason:"),
            html.P(reason, className="text-muted")
        ])
    ])

# Separate callbacks for each problem type to handle dynamic components
@app.callback(
    Output('strategy-recommendation', 'children', allow_duplicate=True),
    [Input('sorting-data-size', 'value'),
     Input('sorting-data-type', 'value'),
     Input('memory-constraint', 'value')],
    prevent_initial_call=True
)
def update_sorting_recommendation(data_size, data_type, memory_constraint):
    """Update recommendation specifically for sorting problems."""
    if data_size is None:
        return dash.no_update
    
    # Provide specific recommendations based on actual parameter values
    if data_size < 50:
        recommendation = "**Insertion Sort** - Simple and efficient for small datasets"
        reason = f"For small datasets ({data_size} elements), simple algorithms like Insertion Sort often outperform complex ones due to lower overhead. Time complexity: O(n²) but very fast for small n."
    elif data_type == 'nearly-sorted':
        recommendation = "**Bubble Sort** - Excellent for nearly sorted data"
        reason = f"Bubble Sort can be very efficient when data is nearly sorted, often achieving O(n) performance instead of O(n²). Perfect for data that's already mostly in order."
    elif memory_constraint:
        recommendation = "**Selection Sort** - In-place sorting with minimal memory"
        reason = f"Selection Sort uses constant extra space O(1) regardless of input size ({data_size} elements). Ideal when memory is severely constrained."
    elif data_type == 'duplicates':
        recommendation = "**Quick Sort with 3-way partitioning** - Efficient for data with many duplicates"
        reason = f"Standard Quick Sort can degrade to O(n²) with many duplicates. 3-way partitioning handles duplicates efficiently, maintaining O(n log n) average performance."
    elif data_size > 10000:
        recommendation = "**Merge Sort** - Guaranteed performance for large datasets"
        reason = f"For large datasets ({data_size} elements), Merge Sort's guaranteed O(n log n) performance and stability make it a reliable choice."
    else:
        recommendation = "**Quick Sort** - Fast average-case performance"
        reason = f"For typical datasets ({data_size} elements), Quick Sort's average O(n log n) performance and in-place operation make it an excellent choice."
    
    return dbc.Card([
        dbc.CardHeader("Recommendation"),
        dbc.CardBody([
            html.H5("Recommendation:"),
            html.P(recommendation, className="font-weight-bold text-primary"),
            html.H5("Reason:"),
            html.P(reason, className="text-muted")
        ])
    ])

@app.callback(
    Output('strategy-recommendation', 'children', allow_duplicate=True),
    [Input('data-status', 'value'),
     Input('search-frequency', 'value')],
    prevent_initial_call=True
)
def update_searching_recommendation(data_status, search_frequency):
    """Update recommendation specifically for searching problems."""
    if data_status is None:
        return dash.no_update
    
    if data_status == 'sorted':
        recommendation = "**Binary Search** - Optimal for sorted data"
        reason = "Binary Search provides O(log n) performance on sorted data, making it the optimal choice for searching in sorted arrays."
    elif search_frequency == 'frequent':
        recommendation = "**Sort first, then Binary Search** - Transform and Conquer"
        reason = "For frequent searches, the O(n log n) cost of sorting is amortized over multiple O(log n) searches. More efficient than repeated O(n) sequential searches."
    else:
        recommendation = "**Sequential Search** - Simple and no preprocessing needed"
        reason = "For one-time searches on unsorted data, Sequential Search is straightforward and doesn't require preprocessing."
    
    return dbc.Card([
        dbc.CardHeader("Recommendation"),
        dbc.CardBody([
            html.H5("Recommendation:"),
            html.P(recommendation, className="font-weight-bold text-primary"),
            html.H5("Reason:"),
            html.P(reason, className="text-muted")
        ])
    ])

@app.callback(
    Output('strategy-recommendation', 'children', allow_duplicate=True),
    [Input('problem-nature', 'value')],
    prevent_initial_call=True
)
def update_optimization_recommendation(problem_nature):
    """Update recommendation specifically for optimization problems."""
    if problem_nature is None:
        return dash.no_update
    
    if problem_nature == 'greedy-works':
        recommendation = "**Greedy Algorithm** - Fast and often optimal"
        reason = "When greedy choices lead to optimal solutions, these algorithms are very efficient and straightforward to implement."
    elif problem_nature == 'greedy-fails':
        recommendation = "**Dynamic Programming or Branch and Bound** - Guaranteed optimal"
        reason = "When greedy fails, we need algorithms that explore multiple solution paths to guarantee optimal solutions."
    else:
        recommendation = "**Try Greedy First, then Dynamic Programming** - Empirical approach"
        reason = "Start with greedy for speed, then use more sophisticated methods if needed. This approach balances efficiency with optimality."
    
    return dbc.Card([
        dbc.CardHeader("Recommendation"),
        dbc.CardBody([
            html.H5("Recommendation:"),
            html.P(recommendation, className="font-weight-bold text-primary"),
            html.H5("Reason:"),
            html.P(reason, className="text-muted")
        ])
    ])

# Callback for toggling algorithm code visibility
@app.callback(
    [Output('toggle-code-btn', 'children'),
     Output('algorithm-code-content', 'children'),
     Output('algorithm-code-content', 'style')],
    [Input('toggle-code-btn', 'n_clicks')]
)
def toggle_algorithm_code(n_clicks: int) -> Tuple[str, html.Div, Dict[str, str]]:
    """Toggle the visibility of algorithm code."""
    if n_clicks is None or n_clicks == 0:
        # Default state: hidden
        return "Show Code", html.Div(), {'display': 'none'}
    
    if n_clicks % 2 == 1:
        # Show code
        code_content = html.Div([
            # Sequential Search Code
            html.Div([
                html.H6("Sequential Search (Brute Force)", className="text-primary mb-2"),
                html.Pre("""
def sequential_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1
                """.strip(), className="bg-light p-2 rounded font-monospace small"),
                html.P("Time Complexity: O(n)", className="text-muted small mt-1")
            ], className="mb-3"),
            
            # Binary Search Code
            html.Div([
                html.H6("Binary Search (Decrease & Conquer)", className="text-success mb-2"),
                html.Pre("""
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
                """.strip(), className="bg-light p-2 rounded font-monospace small"),
                html.P("Time Complexity: O(log n)", className="text-muted small mt-1")
            ])
        ])
        return "Hide Code", code_content, {'display': 'block'}
    else:
        # Hide code
        return "Show Code", html.Div(), {'display': 'none'}

# Callbacks for coin change dashboard
@app.callback(
    [Output('coin-change-results', 'children'),
     Output('coin-change-graph', 'figure')],
    [Input('change-amount', 'value'),
     Input('coin-system', 'value')]
)
def update_coin_change(amount: int, coin_system: str) -> Tuple[html.Div, go.Figure]:
    """Update coin change results based on parameters."""
    if coin_system == 'us':
        coins = [1, 5, 10, 25]
    elif coin_system == 'problematic':
        coins = [1, 10, 25]
    else:
        coins = [1, 5, 10, 20, 25]
    
    # Calculate solutions
    greedy_solution, greedy_count = greedy_coin_change(amount, coins)
    optimal_solution, optimal_count = optimal_coin_change(amount, coins)
    
    # Create results display
    results = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Greedy Solution"),
                dbc.CardBody([
                    html.P(f"Coins used: {greedy_solution}"),
                    html.P(f"Total coins: {greedy_count}")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Optimal Solution"),
                dbc.CardBody([
                    html.P(f"Coins used: {optimal_solution}"),
                    html.P(f"Total coins: {optimal_count}")
                ])
            ])
        ], width=6)
    ])
    
    # Add result message
    if greedy_count == optimal_count:
        alert = dbc.Alert("✅ Greedy algorithm found the optimal solution!", 
                         color="success", className="mt-3")
    else:
        alert = dbc.Alert(f"❌ Greedy algorithm failed! Used {greedy_count} coins instead of {optimal_count}", 
                         color="danger", className="mt-3")
    
    results = dbc.Container([results, alert])
    
    # Create comparison chart
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
        height=300,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    return results, fig

# Callbacks for complexity dashboard
@app.callback(
    [Output('complexity-graph', 'figure'),
     Output('performance-table', 'children')],
    [Input('complexity-functions', 'value'),
     Input('max-n', 'value')]
)
def update_complexity_graph(selected_functions: List[str], max_n: int) -> Tuple[go.Figure, html.Div]:
    """Update the complexity comparison graph."""
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
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    # Create performance table
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
    table = dbc.Table.from_dataframe(
        df, 
        striped=True, 
        bordered=True, 
        hover=True,
        className="mt-3"
    )
    
    return fig, table

# Callbacks for strategy dashboard
@app.callback(
    Output('problem-specific-controls', 'children'),
    Input('problem-type', 'value')
)
def update_problem_controls(problem_type: str) -> html.Div:
    """Update controls based on problem type."""
    if problem_type == 'sorting':
        return html.Div([
            html.Div([
                dbc.Label("Data Size:"),
                dcc.Slider(
                    id='sorting-data-size', 
                    min=10, max=1000000, value=1000, 
                    step=1,
                    marks={i: str(i) for i in [10, 100, 1000, 10000, 100000, 1000000]},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], className="mb-3"),
            html.Div([
                dbc.Label("Data Characteristics:"),
                dcc.Dropdown(
                    id='sorting-data-type',
                    options=[
                        {'label': 'Random', 'value': 'random'},
                        {'label': 'Nearly Sorted', 'value': 'nearly-sorted'},
                        {'label': 'Reversed', 'value': 'reversed'},
                        {'label': 'Many Duplicates', 'value': 'duplicates'}
                    ],
                    value='random'
                )
            ], className="mb-3"),
            dbc.Checklist(id='memory-constraint', options=[{'label': 'Memory Constraint', 'value': True}])
        ])
    elif problem_type == 'searching':
        return html.Div([
            html.Div([
                dbc.Label("Data Size:"),
                dcc.Slider(
                    id='searching-data-size', 
                    min=10, max=1000000, value=1000, 
                    step=1,
                    marks={i: str(i) for i in [10, 100, 1000, 10000, 100000, 1000000]},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], className="mb-3"),
            html.Div([
                dbc.Label("Data Status:"),
                dcc.Dropdown(
                    id='data-status',
                    options=[
                        {'label': 'Unsorted', 'value': 'unsorted'},
                        {'label': 'Sorted', 'value': 'sorted'}
                    ],
                    value='unsorted'
                )
            ], className="mb-3"),
            html.Div([
                dbc.Label("Search Frequency:"),
                dcc.Dropdown(
                    id='search-frequency',
                    options=[
                        {'label': 'One-time', 'value': 'one-time'},
                        {'label': 'Frequent', 'value': 'frequent'}
                    ],
                    value='one-time'
                )
            ], className="mb-3")
        ])
    elif problem_type == 'optimization':
        return html.Div([
            html.Div([
                dbc.Label("Problem Nature:"),
                dcc.Dropdown(
                    id='problem-nature',
                    options=[
                        {'label': 'Greedy Choice Works', 'value': 'greedy-works'},
                        {'label': 'Greedy Choice Fails', 'value': 'greedy-fails'},
                        {'label': 'Unknown', 'value': 'unknown'}
                    ],
                    value='unknown'
                )
            ], className="mb-3"),
            html.Div([
                dbc.Label("Problem Size:"),
                dcc.Slider(
                    id='optimization-size', 
                    min=10, max=10000, value=100, 
                    step=1,
                    marks={i: str(i) for i in [10, 100, 1000, 10000]},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], className="mb-3")
        ])
    else:
        return html.Div("Select a problem type to see specific controls.")

@app.callback(
    [Output('strategy-recommendation', 'children'),
     Output('strategy-comparison-table', 'children')],
    [Input('problem-type', 'value'),
     Input('problem-specific-controls', 'children')]
)
def update_strategy_recommendation(problem_type: str, controls_children) -> Tuple[html.Div, html.Div]:
    """Update strategy recommendation based on problem parameters."""
    
    # Initial recommendation based on problem type
    if problem_type == 'sorting':
        recommendation = "**Quick Sort or Merge Sort** - Recommended for typical datasets"
        reason = "For most sorting scenarios, Quick Sort (average O(n log n)) or Merge Sort (guaranteed O(n log n)) provide excellent performance. Quick Sort is often faster in practice, while Merge Sort guarantees consistent performance."
    
    elif problem_type == 'searching':
        recommendation = "**Binary Search** - Optimal for sorted data"
        reason = "Binary Search provides O(log n) performance on sorted data, making it the optimal choice for searching in sorted arrays."
    
    elif problem_type == 'optimization':
        recommendation = "**Try Greedy First, then Dynamic Programming** - Empirical approach"
        reason = "Start with greedy for speed, then use more sophisticated methods if needed. This approach balances efficiency with optimality."
    
    else:
        recommendation = "Select a problem type to get a recommendation."
        reason = ""
    
    recommendation_div = dbc.Card([
        dbc.CardHeader("Recommendation"),
        dbc.CardBody([
            html.H5("Recommendation:"),
            html.P(recommendation, className="font-weight-bold text-primary"),
            html.H5("Reason:"),
            html.P(reason, className="text-muted")
        ])
    ])
    
    # Strategy comparison table
    strategies_data = {
        "Strategy": ["Brute Force", "Divide & Conquer", "Decrease & Conquer", "Transform & Conquer", "Greedy"],
        "Best For": ["Small problems", "Large problems", "Search problems", "Preprocessing helps", "Optimization"],
        "Complexity": ["Often O(n²) or worse", "Often O(n log n)", "Often O(log n)", "Varies", "Often O(n log n)"],
        "Guarantee": ["Always works", "Often optimal", "Often optimal", "Depends on transform", "May not be optimal"]
    }
    
    df = pd.DataFrame(strategies_data)
    table = dbc.Table.from_dataframe(
        df, 
        striped=True, 
        bordered=True, 
        hover=True,
        className="mt-3"
    )
    
    return recommendation_div, table

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8051)
