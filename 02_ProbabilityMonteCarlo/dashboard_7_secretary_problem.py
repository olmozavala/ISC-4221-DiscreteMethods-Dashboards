"""
Interactive Dashboard 7: Secretary Problem Simulator
Topic: Optimal stopping and decision-making
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple, Dict, Any
import pandas as pd
from dash.exceptions import PreventUpdate

# Initialize the Dash app with Bootstrap
app = dash.Dash(__name__, title="Secretary Problem Simulator", 
                external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

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
        name='Success Rate',
        yaxis='y',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Average rank (secondary y-axis)
    fig.add_trace(go.Scatter(
        x=look_percentages,
        y=avg_ranks,
        mode='lines+markers',
        name='Average Rank',
        yaxis='y2',
        line=dict(color='red', width=2),
        marker=dict(size=6)
    ))
    
    # Optimal strategy line (1/e ≈ 37%)
    optimal_percentage = 100 / np.e  # ≈ 36.8%
    fig.add_vline(x=optimal_percentage, line_dash="dash", line_color="green",
                  annotation_text=f"Optimal: {optimal_percentage:.1f}%", 
                  annotation_position="top right")
    
    fig.update_layout(
        title="Performance vs Look Percentage",
        xaxis_title="Look Percentage (%)",
        yaxis=dict(title="Success Rate", side="left"),
        yaxis2=dict(title="Average Rank", side="right", overlaying="y"),
        showlegend=True,
        height=400
    )
    
    return fig

def create_applicant_plot(applicants: List[float], hired_position: int, 
                         benchmark: float, best_score: float) -> go.Figure:
    """Create plot showing applicant scores and decision points."""
    fig = go.Figure()
    
    # Applicant scores
    fig.add_trace(go.Scatter(
        x=list(range(1, len(applicants) + 1)),
        y=applicants,
        mode='lines+markers',
        name='Applicant Scores',
        line=dict(color='lightblue', width=1),
        marker=dict(size=6)
    ))
    
    # Benchmark line
    fig.add_hline(y=benchmark, line_dash="dash", line_color="orange",
                  annotation_text=f"Benchmark: {benchmark:.3f}")
    
    # Best score line
    fig.add_hline(y=best_score, line_dash="dash", line_color="green",
                  annotation_text=f"Best Score: {best_score:.3f}")
    
    # Hired position
    if hired_position <= len(applicants):
        hired_score = applicants[hired_position - 1]
        fig.add_trace(go.Scatter(
            x=[hired_position],
            y=[hired_score],
            mode='markers',
            name='Hired',
            marker=dict(color='red', size=12, symbol='star')
        ))
    
    fig.update_layout(
        title="Applicant Scores and Decision",
        xaxis_title="Applicant Position",
        yaxis_title="Quality Score",
        showlegend=True,
        height=400
    )
    
    return fig

def create_rank_distribution(results: List[Dict[str, Any]]) -> go.Figure:
    """Create histogram of hired candidate ranks."""
    ranks = [r['rank'] for r in results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=ranks,
        nbinsx=min(20, max(ranks)),
        name='Rank Distribution',
        marker_color='lightblue'
    ))
    
    # Add mean rank line
    mean_rank = np.mean(ranks)
    fig.add_vline(x=mean_rank, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_rank:.1f}")
    
    fig.update_layout(
        title="Distribution of Hired Candidate Ranks",
        xaxis_title="Rank (1 = Best)",
        yaxis_title="Frequency",
        showlegend=True,
        height=400
    )
    
    return fig

# App layout
app.layout = dbc.Container([
    # Hidden divs for storing state
    html.Div(id='simulation-results', style={'display': 'none'}),
    html.Div(id='performance-data', style={'display': 'none'}),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("👔 Secretary Problem Simulator", className="text-center mb-4"),
            html.P("Explore optimal stopping strategies in decision-making", 
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
                    html.Label("Number of Applicants:", className="form-label"),
                    dcc.Slider(
                        id='n-applicants-slider',
                        min=10,
                        max=100,
                        step=10,
                        value=50,
                        marks={10: '10', 25: '25', 50: '50', 75: '75', 100: '100'},
                        className="mb-3"
                    ),
                    
                    html.Label("Look Percentage (%):", className="form-label"),
                    dcc.Slider(
                        id='look-percentage-slider',
                        min=0,
                        max=50,
                        step=5,
                        value=37,
                        marks={0: '0%', 10: '10%', 20: '20%', 30: '30%', 37: '37%', 50: '50%'},
                        className="mb-3"
                    ),
                    
                    html.Label("Number of Simulations:", className="form-label"),
                    dcc.Slider(
                        id='n-simulations-slider',
                        min=100,
                        max=10000,
                        step=100,
                        value=1000,
                        marks={100: '100', 1000: '1K', 5000: '5K', 10000: '10K'},
                        className="mb-3"
                    ),
                    
                    html.Hr(),
                    
                    dbc.Button("Run Single Simulation", id="run-single-btn", 
                              color="primary", className="w-100 mb-2"),
                    
                    dbc.Button("Run Multiple Simulations", id="run-multiple-btn", 
                              color="success", className="w-100 mb-2"),
                    
                    dbc.Button("Performance Analysis", id="performance-btn", 
                              color="info", className="w-100")
                ])
            ], className="h-100")
        ], width=3),
        
        # Main content area
        dbc.Col([
            # Navigation tabs
            dbc.Tabs([
                dbc.Tab([
                    dbc.Row([
                        # Single simulation results
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("📊 Single Simulation Results", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    html.Div(id='single-simulation-results')
                                ])
                            ])
                        ], width=6),
                        
                        # Multiple simulation statistics
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("📈 Multiple Simulation Statistics", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    html.Div(id='multiple-simulation-stats')
                                ])
                            ])
                        ], width=6)
                    ], className="mb-3"),
                    
                    # Applicant plot
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("👥 Applicant Scores", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    dcc.Graph(id='applicant-plot')
                                ])
                            ])
                        ])
                    ], className="mb-3"),
                    
                    # Rank distribution
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("📊 Rank Distribution", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    dcc.Graph(id='rank-distribution-plot')
                                ])
                            ])
                        ])
                    ], className="mb-3")
                ], label="Simulation Results", tab_id="tab-results"),
                
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("📈 Performance Analysis", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    dcc.Graph(id='performance-plot')
                                ])
                            ])
                        ])
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader([
                                    html.H5("📊 Optimal Strategy Analysis", className="mb-0")
                                ]),
                                dbc.CardBody([
                                    html.Div(id='optimal-strategy-content')
                                ])
                            ])
                        ])
                    ])
                ], label="Performance Analysis", tab_id="tab-performance")
            ], id="tabs", active_tab="tab-results"),
            
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
                                html.Li("Optimal Stopping: When to stop searching and make a decision"),
                                html.Li("Look-then-Leap Strategy: Observe first k candidates, then hire the first better one"),
                                html.Li("1/e Rule: Optimal to look at first 37% of candidates"),
                                html.Li("Success Rate: Probability of hiring the best candidate"),
                                html.Li("Expected Rank: Average quality of hired candidate")
                            ]),
                            html.Hr(),
                            html.H6("Mathematical Foundation:"),
                            html.P("The optimal strategy is to reject the first n/e candidates and then hire the first one better than all seen so far.")
                        ])
                    ])
                ])
            ])
        ], width=9)
    ])
], fluid=True)

# Callbacks
@app.callback(
    [Output('single-simulation-results', 'children'),
     Output('applicant-plot', 'figure')],
    [Input('run-single-btn', 'n_clicks')],
    [State('n-applicants-slider', 'value'),
     State('look-percentage-slider', 'value')]
)
def run_single_simulation(n_clicks: int, n_applicants: int, look_percentage: float) -> Tuple[html.Div, go.Figure]:
    """Run single simulation and display results."""
    if n_clicks is None:
        raise PreventUpdate
    
    # Generate applicants and run simulation
    applicants = generate_applicant_pool(n_applicants)
    result = simulate_secretary_problem(applicants, look_percentage)
    
    # Create results content
    results_content = html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{result['hired_score']:.3f}", className="card-title"),
                        html.P("Hired Score", className="card-text")
                    ])
                ], color="success", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{result['rank']}", className="card-title"),
                        html.P("Rank (1=Best)", className="card-text")
                    ])
                ], color="info", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{result['hired_position']}", className="card-title"),
                        html.P("Hired Position", className="card-text")
                    ])
                ], color="warning", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("✅" if result['is_best'] else "❌", className="card-title"),
                        html.P("Best Candidate", className="card-text")
                    ])
                ], color="primary", outline=True)
            ], width=3)
        ]),
        html.Hr(),
        html.P(f"Benchmark Score: {result['benchmark']:.3f}"),
        html.P(f"Best Available Score: {result['best_score']:.3f}"),
        html.P(f"Look Percentage: {look_percentage}% ({int(n_applicants * look_percentage / 100)} candidates)")
    ])
    
    # Create applicant plot
    applicant_fig = create_applicant_plot(applicants, result['hired_position'], 
                                        result['benchmark'], result['best_score'])
    
    return results_content, applicant_fig

@app.callback(
    [Output('multiple-simulation-stats', 'children'),
     Output('rank-distribution-plot', 'figure')],
    [Input('run-multiple-btn', 'n_clicks')],
    [State('n-applicants-slider', 'value'),
     State('look-percentage-slider', 'value'),
     State('n-simulations-slider', 'value')]
)
def run_multiple_simulations_callback(n_clicks: int, n_applicants: int, 
                                    look_percentage: float, n_simulations: int) -> Tuple[html.Div, go.Figure]:
    """Run multiple simulations and display statistics."""
    if n_clicks is None:
        raise PreventUpdate
    
    # Run multiple simulations
    results = run_multiple_simulations(n_applicants, look_percentage, n_simulations)
    stats = calculate_statistics(results)
    
    # Create statistics content
    stats_content = html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{stats['success_rate']:.1%}", className="card-title"),
                        html.P("Success Rate", className="card-text")
                    ])
                ], color="success", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{stats['avg_rank']:.1f}", className="card-title"),
                        html.P("Average Rank", className="card-text")
                    ])
                ], color="info", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{stats['avg_score']:.3f}", className="card-title"),
                        html.P("Average Score", className="card-text")
                    ])
                ], color="warning", outline=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{stats['avg_position']:.1f}", className="card-title"),
                        html.P("Avg Position", className="card-text")
                    ])
                ], color="primary", outline=True)
            ], width=3)
        ]),
        html.Hr(),
        html.P(f"Total Simulations: {n_simulations:,}"),
        html.P(f"Look Percentage: {look_percentage}%"),
        html.P(f"Number of Applicants: {n_applicants}")
    ])
    
    # Create rank distribution plot
    rank_fig = create_rank_distribution(results)
    
    return stats_content, rank_fig

@app.callback(
    [Output('performance-plot', 'figure'),
     Output('optimal-strategy-content', 'children')],
    [Input('performance-btn', 'n_clicks')],
    [State('n-applicants-slider', 'value'),
     State('n-simulations-slider', 'value')]
)
def run_performance_analysis(n_clicks: int, n_applicants: int, n_simulations: int) -> Tuple[go.Figure, html.Div]:
    """Run performance analysis across different look percentages."""
    if n_clicks is None:
        raise PreventUpdate
    
    # Test different look percentages
    look_percentages = list(range(0, 51, 5))  # 0% to 50% in steps of 5%
    success_rates = []
    avg_ranks = []
    
    for look_pct in look_percentages:
        results = run_multiple_simulations(n_applicants, look_pct, n_simulations)
        stats = calculate_statistics(results)
        success_rates.append(stats['success_rate'])
        avg_ranks.append(stats['avg_rank'])
    
    # Create performance plot
    performance_fig = create_performance_plot(look_percentages, success_rates, avg_ranks)
    
    # Find optimal strategy
    optimal_idx = np.argmax(success_rates)
    optimal_percentage = look_percentages[optimal_idx]
    optimal_success_rate = success_rates[optimal_idx]
    theoretical_optimal = 100 / np.e  # ≈ 36.8%
    
    # Create optimal strategy content
    optimal_content = html.Div([
        html.H6("Optimal Strategy Analysis:"),
        html.P(f"Simulated Optimal: {optimal_percentage}% (Success Rate: {optimal_success_rate:.1%})"),
        html.P(f"Theoretical Optimal: {theoretical_optimal:.1f}% (1/e rule)"),
        html.P(f"Difference: {abs(optimal_percentage - theoretical_optimal):.1f}%"),
        html.Hr(),
        html.H6("Strategy Comparison:"),
        html.P(f"Random Selection: ~{1/n_applicants:.1%} success rate"),
        html.P(f"Always Pick First: ~{1/n_applicants:.1%} success rate"),
        html.P(f"Optimal Strategy: ~{optimal_success_rate:.1%} success rate"),
        html.Hr(),
        html.H6("Key Insights:"),
        html.Ul([
            html.Li("The 1/e rule (≈37%) is approximately optimal"),
            html.Li("Too little looking: miss good candidates early"),
            html.Li("Too much looking: run out of candidates"),
            html.Li("Balance between exploration and exploitation")
        ])
    ])
    
    return performance_fig, optimal_content

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8053)


