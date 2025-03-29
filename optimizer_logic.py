# optimizer_logic.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Import graph_objects for more control
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
# Removed io and contextlib as they weren't strictly needed in the last version

# --- Data Fetching (no changes) ---
def fetch_data(file_path):
    """Loads stock data from a CSV file."""
    try:
        data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data.dropna(axis=1, how='any', inplace=True)
        if data.empty:
            raise ValueError("No valid numeric data found after cleaning.")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# --- Portfolio Optimization (no changes) ---
def optimize_portfolio(data):
    """
    Performs portfolio optimization using PyPortfolioOpt.
    Returns: tuple: (cleaned_weights, performance_dict, sharpe_evaluation) or (None, None, None) if error.
    """
    if data is None or data.empty:
        print("Error: Input data is invalid for optimization.")
        return None, None, None
    try:
        mu = mean_historical_return(data)
        S = CovarianceShrinkage(data).ledoit_wolf()
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        expected_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=False)
        performance_dict = {
            "Expected annual return": expected_return,
            "Annual volatility": annual_volatility,
            "Sharpe Ratio": sharpe_ratio
        }
        if sharpe_ratio is None or np.isnan(sharpe_ratio): sharpe_evaluation = 'Unavailable'
        elif sharpe_ratio > 3.0: sharpe_evaluation = 'Exceptional ✨'
        elif sharpe_ratio > 2.0: sharpe_evaluation = 'Excellent 👍'
        elif sharpe_ratio > 1.0: sharpe_evaluation = 'Good 🙂'
        else: sharpe_evaluation = 'Suboptimal 🤔'
        return cleaned_weights, performance_dict, sharpe_evaluation
    except ValueError as ve:
         print(f"Optimization Error: {ve}. Check if data has sufficient history or variance.")
         return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred during optimization: {e}")
        return None, None, None

# --- Discrete Allocation (no changes) ---
def calculate_allocation(weights, data, total_portfolio_value):
    """Calculates discrete share allocation."""
    if weights is None or data is None or data.empty:
         print("Error: Invalid input for allocation.")
         return None, None, None
    try:
        latest_prices = get_latest_prices(data)
        da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)
        allocation, leftover = da.lp_portfolio()
        return allocation, leftover, latest_prices
    except Exception as e:
        print(f"Error during discrete allocation: {e}")
        return None, None, None


# --- Plotting (MODIFIED FOR SLEEK "APPLE-ESQUE" STYLE) ---
def create_weights_plot(weights):
    """Creates a sleek, Apple-inspired Plotly figure of portfolio weights."""
    if not weights:
        print("No weights to plot.")
        return None

    # Convert weights dictionary to DataFrame and sort for better visualization
    df_weights = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])
    df_weights = df_weights.sort_values(by='Weight', ascending=False)
    # Filter out zero weights AFTER sorting if needed (optional)
    df_weights = df_weights[df_weights['Weight'] > 0.0001] # Threshold to remove tiny weights


    # Use Plotly Graph Objects for fine-grained control
    fig = go.Figure()

    # --- Define the aesthetic elements ---
    # Subtle, sophisticated color (e.g., desaturated blue, graphite gray)
    # bar_color = '#6495ED' # Cornflower Blue (softer than cyan)
    # bar_color = '#888888' # Medium Gray
    bar_color = '#4A90E2' # A slightly more saturated, professional blue

    # You could even use a subtle gradient if desired (more complex)
    # For simplicity, we'll stick to a single refined color

    fig.add_trace(go.Bar(
        x=df_weights['Asset'],
        y=df_weights['Weight'],
        name='Weights',
        marker=dict(
            color=bar_color,
            line=dict(
                color=bar_color, # Make border same as fill
                width=1
            )
        ),
        # Rounded corners aren't directly supported in go.Bar in a simple way.
        # We achieve sleekness through color, spacing, and minimalism instead.
        text=[f'{w:.1%}' for w in df_weights['Weight']], # Clean percentage format (1 decimal)
        textposition='outside',
        textfont=dict(
            size=10, # Keep text size reasonable
            color='#CCCCCC' # Lighter grey for text labels, less harsh than pure white
        ),
        hoverinfo='x+y', # Basic hover info
        hovertemplate='<b>%{x}</b><br>Weight: %{y:.2%}<extra></extra>' # Cleaner hover text
    ))

    # --- Customize layout for the sleek, minimalist aesthetic ---
    fig.update_layout(
        title=dict(
            text='Optimized Portfolio Allocation', # Slightly different title
            x=0.5, # Center title
            xanchor='center',
            font=dict(
                size=18,
                color='#F0F0F0' # Off-white for title
                # family="Your Preferred Sans-Serif Font" # Font family (best effort)
            )
        ),
        xaxis_title=None, # No x-axis title for cleaner look
        yaxis_title="Weight (%)", # Add units to y-axis title
        yaxis_tickformat='.0%', # Format y-axis ticks as percentages
        plot_bgcolor='rgba(0,0,0,0)', # Fully transparent plot background
        paper_bgcolor='rgba(0,0,0,0)', # Fully transparent paper background
        font=dict( # Global font settings
            color='#CCCCCC', # Light grey for general text elements
            # family="Your Preferred Sans-Serif Font"
        ),
        yaxis=dict(
            showgrid=True, # Keep grid lines for reference
            gridcolor='#444444', # Darker, more subtle grid lines
            gridwidth=1,
            zeroline=False, # Hide the y=0 line
            showline=False, # Hide the main y-axis line
            tickfont=dict(color='#AAAAAA'), # Grey for axis tick labels
            title_font=dict(color='#AAAAAA', size=12) # Style y-axis title
        ),
        xaxis=dict(
            showgrid=False, # No grid on x-axis
            showline=True, # Show the x-axis line for structure
            linecolor='#666666', # Subtle grey for x-axis line
            linewidth=1,
            tickfont=dict(color='#AAAAAA'), # Grey for axis tick labels
            tickangle=-45 # Keep rotation for readability if many assets
        ),
        margin=dict(l=60, r=30, t=70, b=120), # Adjust margins for better spacing (esp. bottom)
        bargap=0.2, # Standard gap, adjust if needed (0.15 for tighter)
        # hovermode='x unified', # Alternative hover: shows all bars at that x-position
        hovermode='closest', # Default hover: highlights the closest bar
        showlegend=False # No legend needed for a single trace bar chart
    )

    return fig