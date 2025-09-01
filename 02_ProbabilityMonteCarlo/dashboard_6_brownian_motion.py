"""
Interactive Dashboard 6: Brownian Motion Simulator
Topic: Stochastic processes and random walks
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple, Dict
import pandas as pd
from dash.exceptions import PreventUpdate

# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__, title="Brownian Motion Simulator", 
                external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

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

def calculate_statistics(positions: List[List[float]], time: List[float], volatility: float = 1.0) -> Dict[str, float]:
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
            
            # Add starting point
            fig.add_trace(go.Scatter(
                x=[x_pos[0]], y=[y_pos[0]],
                mode='markers',
                marker=dict(color=color, size=8, symbol='circle'),
                showlegend=False,
                hovertemplate='Start<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
            ))
            
            # Add ending point
            fig.add_trace(go.Scatter(
                x=[x_pos[-1]], y=[y_pos[-1]],
                mode='markers',
                marker=dict(color=color, size=8, symbol='star'),
                showlegend=False,
                hovertemplate='End<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
            ))
    else:
        # Single particle
        x_pos = x_positions[0]
        y_pos = y_positions[0]
        
        fig.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='lines',
            name='Particle Path',
            line=dict(color='blue', width=2),
            hovertemplate='Time: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
            text=[f'{t:.2f}' for t in time]
        ))
        
        # Add starting and ending points
        fig.add_trace(go.Scatter(
            x=[x_pos[0]], y=[y_pos[0]],
            mode='markers',
            marker=dict(color='green', size=10, symbol='circle'),
            name='Start',
            hovertemplate='Start<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[x_pos[-1]], y=[y_pos[-1]],
            mode='markers',
            marker=dict(color='red', size=10, symbol='star'),
            name='End',
            hovertemplate='End<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="2D Brownian Motion Path",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        showlegend=True,
        height=500
    )
    
    return fig

def create_time_series_plot(simulation_data: Dict[str, List], component: str = 'x') -> go.Figure:
    """Create time series plot for x or y component."""
    fig = go.Figure()
    
    time = simulation_data['time']
    positions = simulation_data[f'{component}_positions']
    
    colors = px.colors.qualitative.Set1
    for i, pos in enumerate(positions):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=time, y=pos,
            mode='lines',
            name=f'Particle {i+1}',
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        title=f"{component.upper()}-Position vs Time",
        xaxis_title="Time",
        yaxis_title=f"{component.upper()}-Position",
        showlegend=True,
        height=400
    )
    
    return fig

def create_stock_price_plot(simulation_data: Dict[str, List]) -> go.Figure:
    """Create stock price plot for geometric Brownian motion."""
    fig = go.Figure()
    
    time = simulation_data['time']
    prices = simulation_data['prices']
    
    fig.add_trace(go.Scatter(
        x=time, y=prices,
        mode='lines',
        name='Stock Price',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Geometric Brownian Motion - Stock Price",
        xaxis_title="Time",
        yaxis_title="Price",
        showlegend=True,
        height=400
    )
    
    return fig

# App layout
app.layout = dbc.Container([
    # Hidden divs for storing state
    html.Div(id='simulation-data', style={'display': 'none'}),
    html.Div(id='gbm-data', style={'display': 'none'}),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🌊 Brownian Motion Simulator", className="text-center mb-4"),
            html.P("Explore stochastic processes and random walks", 
                   className="text-center text-muted")
        ])
    ]),
    
    # Main content
    dbc.Row([
        # Sidebar
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("⚙️ Simulation Parameters", className="mb-0")
                ]),
                dbc.CardBody([
                    html.Label("Number of Steps:", className="form-label"),
                    dcc.Slider(
                        id='n-steps-slider',
                        min=50,
                        max=1000,
                        step=50,
                        value=200,
                        marks={50: '50', 200: '200', 500: '500', 1000: '1000'},
                        className="mb-3"
                    ),
                    
                    html.Label("Time Step (dt):", className="form-label"),
                    dcc.Slider(
                        id='dt-slider',
                        min=0.01,
                        max=0.1,
                        step=0.01,
                        value=0.05,
                        marks={0.01: '0.01', 0.05: '0.05', 0.1: '0.1'},
                        className="mb-3"
                    ),
                    
                    html.Label("Drift (μ):", className="form-label"),
                    dcc.Slider(
                        id='drift-slider',
                        min=-0.1,
                        max=0.1,
                        step=0.01,
                        value=0.0,
                        marks={-0.1: '-0.1', 0: '0', 0.1: '0.1'},
                        className="mb-3"
                    ),
                    
                    html.Label("Volatility (σ):", className="form-label"),
                    dcc.Slider(
                        id='volatility-slider',
                        min=0.1,
                        max=2.0,
                        step=0.1,
                        value=1.0,
                        marks={0.1: '0.1', 1.0: '1.0', 2.0: '2.0'},
                        className="mb-3"
                    ),
                    
                    html.Label("Number of Particles:", className="form-label"),
                    dcc.Slider(
                        id='n-particles-slider',
                        min=1,
                        max=10,
                        step=1,
                        value=3,
                        marks={i: str(i) for i in range(1, 11)},
                        className="mb-3"
                    ),
                    
                    html.Hr(),
                    
                    dbc.Button("Run Brownian Motion", id="run-bm-btn", 
                              color="primary", className="w-100 mb-2"),
                    
                    dbc.Button("Run Stock Price Simulation", id="run-gbm-btn", 
                              color="success", className="w-100")
                ])
            ], className="h-100")
        ], width=3),
        
        # Main content area
        dbc.Col([
            # Navigation tabs
            dbc.Tabs([
                dbc.Tab([
                    dbc.Row([
                        # Statistics
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("📊 Statistics", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    html.Div(id='statistics-content')
                                ])
                            ])
                        ], width=6),
                        
                        # Parameters
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("⚙️ Current Parameters", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    html.Div(id='parameters-content')
                                ])
                            ])
                        ], width=6)
                    ], className="mb-3"),
                    
                    # Path plot
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("🌊 2D Brownian Motion Path", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    dcc.Graph(id='path-plot')
                                ])
                            ])
                        ])
                    ], className="mb-3"),
                    
                    # Time series plots
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("📈 X-Position vs Time", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    dcc.Graph(id='x-time-plot')
                                ])
                            ])
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("📈 Y-Position vs Time", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    dcc.Graph(id='y-time-plot')
                                ])
                            ])
                        ], width=6)
                    ], className="mb-3")
                ], label="Brownian Motion", tab_id="tab-bm"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("📈 Stock Price Simulation", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    dcc.Graph(id='stock-price-plot')
                                ])
                            ])
                        ])
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("📊 Stock Price Statistics", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    html.Div(id='stock-statistics-content')
                                ])
                            ])
                        ])
                    ])
                ], label="Stock Price (GBM)", tab_id="tab-gbm")
            ], id="tabs", active_tab="tab-bm"),
            
            # Learning objectives
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("🎓 Learning Objectives", className="mb-0")
                        ]),
                        dbc.CardBody([
                            html.H6("Key Concepts:"),
                            html.Ul([
                                html.Li("Brownian Motion: Continuous-time stochastic process"),
                                html.Li("Random Walk: Discrete approximation of Brownian motion"),
                                html.Li("Drift: Systematic trend in the process"),
                                html.Li("Volatility: Measure of randomness/dispersion"),
                                html.Li("Geometric Brownian Motion: Used for modeling stock prices")
                            ]),
                            html.Hr(),
                            html.H6("Mathematical Properties:"),
                            html.P("For standard Brownian motion: E[X(t)] = 0, Var[X(t)] = σ²t")
                        ])
                    ])
                ])
            ])
        ], width=9)
    ])
], fluid=True)

# Callbacks
@app.callback(
    [Output('simulation-data', 'children'),
     Output('statistics-content', 'children'),
     Output('parameters-content', 'children'),
     Output('path-plot', 'figure'),
     Output('x-time-plot', 'figure'),
     Output('y-time-plot', 'figure')],
    [Input('run-bm-btn', 'n_clicks')],
    [State('n-steps-slider', 'value'),
     State('dt-slider', 'value'),
     State('drift-slider', 'value'),
     State('volatility-slider', 'value'),
     State('n-particles-slider', 'value')]
)
def run_brownian_motion(n_clicks: int, n_steps: int, dt: float, drift: float, 
                       volatility: float, n_particles: int) -> Tuple[str, html.Div, html.Div, go.Figure, go.Figure, go.Figure]:
    """Run Brownian motion simulation."""
    if n_clicks is None:
        raise PreventUpdate
    
    # Run simulation
    simulation_data = simulate_brownian_motion(n_steps, dt, drift, volatility, n_particles)
    
    # Calculate statistics
    stats = calculate_statistics(simulation_data['x_positions'], simulation_data['time'], volatility)
    
    # Create statistics content
    stats_content = html.Div([
        html.P(f"Mean Distance: {stats['mean_distance']:.3f}"),
        html.P(f"Theoretical Distance: {stats['theoretical_distance']:.3f}"),
        html.P(f"Standard Deviation: {stats['std_distance']:.3f}"),
        html.P(f"Total Time: {simulation_data['time'][-1]:.2f}"),
        html.P(f"Number of Particles: {n_particles}")
    ])
    
    # Create parameters content
    params_content = html.Div([
        html.P(f"Steps: {n_steps}"),
        html.P(f"Time Step (dt): {dt}"),
        html.P(f"Drift (μ): {drift}"),
        html.P(f"Volatility (σ): {volatility}"),
        html.P(f"Particles: {n_particles}")
    ])
    
    # Create plots
    path_fig = create_path_plot(simulation_data, n_particles > 1)
    x_time_fig = create_time_series_plot(simulation_data, 'x')
    y_time_fig = create_time_series_plot(simulation_data, 'y')
    
    return str(simulation_data), stats_content, params_content, path_fig, x_time_fig, y_time_fig

@app.callback(
    [Output('gbm-data', 'children'),
     Output('stock-price-plot', 'figure'),
     Output('stock-statistics-content', 'children')],
    [Input('run-gbm-btn', 'n_clicks')],
    [State('n-steps-slider', 'value'),
     State('dt-slider', 'value'),
     State('drift-slider', 'value'),
     State('volatility-slider', 'value')]
)
def run_geometric_brownian_motion(n_clicks: int, n_steps: int, dt: float, 
                                 drift: float, volatility: float) -> Tuple[str, go.Figure, html.Div]:
    """Run geometric Brownian motion simulation."""
    if n_clicks is None:
        raise PreventUpdate
    
    # Run simulation
    gbm_data = simulate_geometric_brownian_motion(n_steps, dt, 100.0, drift, volatility)
    
    # Create stock price plot
    stock_fig = create_stock_price_plot(gbm_data)
    
    # Calculate statistics
    prices = gbm_data['prices']
    initial_price = prices[0]
    final_price = prices[-1]
    returns = (final_price - initial_price) / initial_price
    
    stats_content = html.Div([
        html.P(f"Initial Price: ${initial_price:.2f}"),
        html.P(f"Final Price: ${final_price:.2f}"),
        html.P(f"Total Return: {returns:.2%}"),
        html.P(f"Expected Return: {drift * gbm_data['time'][-1]:.2%}"),
        html.P(f"Volatility: {volatility:.2f}"),
        html.P(f"Time Period: {gbm_data['time'][-1]:.2f}")
    ])
    
    return str(gbm_data), stock_fig, stats_content

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8052)


