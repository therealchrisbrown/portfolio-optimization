import streamlit as st
import pandas as pd
import os
from optimizer_logic import (
    fetch_data,
    optimize_portfolio,
    calculate_allocation,
    create_weights_plot # This function now returns a Plotly fig
)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- App Title ---
st.title("✨ Portfolio Optimizer")
st.write("Optimize your stock portfolio using Mean-Variance Optimization (Max Sharpe Ratio).")
st.write("---")

# --- Global Variables ---
DATA_FILE = 'stocks.csv'

# --- Data Loading Section ---
if not os.path.exists(DATA_FILE):
    st.error(f"Error: Data file '{DATA_FILE}' not found in the current directory.")
    st.stop()

data = fetch_data(DATA_FILE)

if data is None:
    st.error("Failed to load or process data. Please check the console for errors and ensure 'stocks.csv' is valid.")
    st.stop()
elif data.empty:
    st.warning("Data loaded but is empty after cleaning. Cannot proceed with optimization.")
    st.stop()
else:
    st.sidebar.subheader("Available Stocks in Dataset")
    st.sidebar.dataframe(data.head(), height=150)
    st.sidebar.caption(f"Loaded {len(data.columns)} assets from '{DATA_FILE}'.")

# --- User Input Section ---
st.header("📊 Configuration")
total_portfolio_value = st.number_input(
    "Enter Total Portfolio Value ($):",
    min_value=100.0,
    max_value=10000000.0,
    value=10000.0,
    step=1000.0,
    help="The total amount of cash you want to invest."
)

# --- Optimization Trigger ---
optimize_button = st.button("🚀 Run Optimization", key="optimize_button")
st.write("---")

# --- Results Section ---
if optimize_button:
    if data is not None and not data.empty:
        st.header("📈 Optimization Results")

        cleaned_weights, performance, sharpe_eval = optimize_portfolio(data)

        if cleaned_weights is not None and performance is not None:
            st.subheader("Portfolio Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Return", f"{performance['Expected annual return']:.2%}")
            col2.metric("Annual Volatility", f"{performance['Annual volatility']:.2%}")
            col3.metric("Sharpe Ratio", f"{performance['Sharpe Ratio']:.2f}", delta=sharpe_eval)

            st.write("---")

            col_plot, col_alloc = st.columns([2, 1])

            with col_plot:
                st.subheader("Optimal Weights")
                weights_fig = create_weights_plot(cleaned_weights)
                if weights_fig:
                    # --- THIS IS THE MODIFIED LINE ---
                    st.plotly_chart(weights_fig, use_container_width=True) # Use st.plotly_chart
                    # --- END MODIFICATION ---
                else:
                    st.warning("Could not generate weights plot.")

            with col_alloc:
                st.subheader("Discrete Allocation")
                allocation, leftover, latest_prices = calculate_allocation(cleaned_weights, data, total_portfolio_value)

                if allocation is not None and latest_prices is not None:
                    alloc_data = []
                    total_invested = 0
                    for stock, shares in allocation.items():
                        cost = shares * latest_prices[stock]
                        total_invested += cost
                        alloc_data.append({
                            "Stock": stock,
                            "Shares": shares,
                            "Latest Price": f"${latest_prices[stock]:.2f}",
                            "Total Cost": f"${cost:.2f}",
                            "Weight": f"{cleaned_weights.get(stock, 0):.2%}"
                            })

                    alloc_df = pd.DataFrame(alloc_data)
                    st.dataframe(alloc_df.set_index('Stock'))

                    st.metric("💰 Total Invested", f"${total_invested:.2f}")
                    st.metric("💸 Leftover Cash", f"${leftover:.2f}")
                else:
                    st.error("Could not calculate discrete allocation.")
        else:
            st.error("Optimization failed. Please check the data or console for more details.")
    else:
        st.warning("Cannot run optimization because data is not loaded correctly.")

# --- Footer/Info ---
st.sidebar.write("---")
st.sidebar.info(
    "Powered by PyPortfolioOpt & Streamlit.\n\n"
    "This app performs portfolio optimization based on historical data."
    "Past performance is not indicative of future results."
)