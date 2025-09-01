# Probability & Monte Carlo Dashboards - Dash Migration

This directory contains interactive dashboards for exploring probability concepts and Monte Carlo methods, now converted from Streamlit to **Dash with Bootstrap components**.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- `uv` package manager (recommended)

### Installation
```bash
# Install dependencies using uv
uv sync

# Or using pip
pip install -r requirements.txt
```

### Running Dashboards

#### Option 1: Use the Dashboard Runner (Recommended)
```bash
uv run python run_dashboards.py
```

This will show a menu where you can:
- Launch individual dashboards
- Launch all dashboards simultaneously
- See available options

#### Option 2: Run Individual Dashboards
```bash
# Probability Building Blocks
uv run python dashboard_1_probability_building_blocks.py

# Monte Carlo π Estimator  
uv run python dashboard_4_monte_carlo_pi.py

# Brownian Motion Simulator
uv run python dashboard_6_brownian_motion.py

# Secretary Problem Simulator
uv run python dashboard_7_secretary_problem.py
```

## 📊 Available Dashboards

| Dashboard | Port | Description |
|-----------|------|-------------|
| **Probability Building Blocks** | 8050 | Explore sample spaces, events, and random variables |
| **Monte Carlo π Estimator** | 8051 | Estimate π using Monte Carlo method |
| **Brownian Motion Simulator** | 8052 | Explore stochastic processes and random walks |
| **Secretary Problem Simulator** | 8053 | Optimal stopping and decision-making |

## 🔄 Migration Summary

### What Changed
- **Framework**: Streamlit → Dash
- **Styling**: Streamlit components → Bootstrap components
- **Architecture**: Session state → Callback-based state management
- **Layout**: Streamlit sidebar → Bootstrap responsive layout

### Key Improvements
1. **Better Performance**: Dash is more efficient for complex interactions
2. **Responsive Design**: Bootstrap components provide better mobile experience
3. **Modular Architecture**: Callback system allows for more complex interactions
4. **Professional Look**: Bootstrap styling provides a more polished appearance
5. **Better State Management**: More control over component updates

### Technical Changes
- Replaced `st.session_state` with hidden `html.Div` elements for state storage
- Converted `st.sidebar` to Bootstrap cards and responsive columns
- Replaced `st.button` with `dbc.Button` components
- Converted `st.slider` to `dcc.Slider` with Bootstrap styling
- Replaced `st.plotly_chart` with `dcc.Graph` components
- Added proper callback functions for interactive updates

## 🎨 UI/UX Enhancements

### Bootstrap Components Used
- `dbc.Container` - Main layout container
- `dbc.Row` and `dbc.Col` - Responsive grid system
- `dbc.Card` - Content organization
- `dbc.Button` - Interactive buttons
- `dbc.Tabs` - Tabbed navigation
- `dbc.Nav` - Navigation components

### Responsive Design
- Mobile-friendly layouts
- Adaptive column sizing
- Touch-friendly controls
- Consistent spacing and typography

## 🔧 Development

### Adding New Dashboards
1. Create a new Python file following the naming convention `dashboard_X_name.py`
2. Use the Dash + Bootstrap template structure
3. Add the dashboard to `run_dashboards.py` configuration
4. Update this README

### Template Structure
```python
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([...]),
    
    # Main content
    dbc.Row([
        # Sidebar
        dbc.Col([...], width=3),
        
        # Content area
        dbc.Col([...], width=9)
    ])
], fluid=True)

# Callbacks
@app.callback(...)
def update_function(...):
    # Implementation
    pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=XXXX)
```

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Check what's using the port
lsof -i :8050

# Kill the process
kill -9 <PID>
```

**Dependencies Missing**
```bash
# Reinstall dependencies
uv sync --reinstall
```

**Dashboard Not Loading**
- Check browser console for JavaScript errors
- Verify all dependencies are installed
- Ensure no firewall blocking the port

### Debug Mode
All dashboards run in debug mode by default. To disable:
```python
app.run(debug=False, host='0.0.0.0', port=XXXX)
```

## 📚 Learning Resources

### Dash Documentation
- [Dash User Guide](https://dash.plotly.com/)
- [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/)
- [Dash Callbacks](https://dash.plotly.com/basic-callbacks)

### Bootstrap Documentation
- [Bootstrap 5](https://getbootstrap.com/docs/5.3/)
- [Bootstrap Components](https://getbootstrap.com/docs/5.3/components/)

## 🤝 Contributing

When contributing to these dashboards:

1. Follow the existing code structure
2. Use type hints for all functions
3. Include docstrings for all functions
4. Follow PEP8 naming conventions
5. Test on different screen sizes
6. Update this README if adding new dashboards

## 📄 License

This project follows the same license as the parent repository. 