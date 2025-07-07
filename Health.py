import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from pyqubo import Array, Constraint, Placeholder
import dynex
import datetime
import seaborn as sns

# Set visual style for plots
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 6)

# Define stock tickers to analyze
stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'INTC', 'AMD', 'IBM']

# Fetch historical data range
today = datetime.datetime.now()
start = today - datetime.timedelta(days=2 * 365)

# Fetch stock prices and returns
def get_price_data(symbols, start, end):
    closing_prices = pd.DataFrame()
    for ticker in symbols:
        data = web.DataReader(ticker, 'stooq', start, end)
        data = data.sort_index()
        closing_prices[ticker] = data['Close']
    returns = closing_prices.pct_change().dropna()
    return closing_prices, returns

# Load and display data
price_data, returns_data = get_price_data(stocks, start, today)
print("Sample Daily Returns:\n", returns_data.head())

# Compute expected returns and covariance matrix
mean_returns = returns_data.mean() * 252
cov_matrix = returns_data.cov() * 252

print("\nAnnualized Returns:\n", mean_returns)
print("\nCovariance Matrix:\n", cov_matrix)

# Heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(returns_data.corr(), annot=True, cmap='viridis', linewidths=0.5)
plt.title('Correlation Between Stock Returns')
plt.tight_layout()
plt.show()

# Plot normalized price movement
norm_prices = price_data / price_data.iloc[0]
plt.figure(figsize=(12, 6))
norm_prices.plot()
plt.title('Stock Performance (Normalized)')
plt.xlabel('Time')
plt.ylabel('Relative Price')
plt.grid(True)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Optimization setup
num_to_pick = 3
risk_aversion = 0.5
constraint_penalty = 0.20

# Minimize negative returns to maximize profit
neg_returns = -mean_returns.values
num_stocks = len(stocks)
x = Array.create('stock', shape=num_stocks, vartype='BINARY')

# Define risk and return objectives
risk_term = sum(cov_matrix.iloc[i, j] * x[i] * x[j] for i in range(num_stocks) for j in range(num_stocks))
return_term = sum(neg_returns[i] * x[i] for i in range(num_stocks))

# Add constraint: select exactly `num_to_pick` stocks
constraint = Constraint(sum(x) - num_to_pick, label='choose_k')

# Final QUBO formulation
H = risk_aversion * risk_term + (1 - risk_aversion) * return_term + constraint_penalty * constraint
model = H.compile()
bqm = model.to_bqm()

# Dynex integration
dnx_model = dynex.BQM(bqm)
sampler = dynex.DynexSampler(dnx_model, mainnet=True, description="Portfolio Optimization via Dynex", bnb=False)
results = sampler.sample(num_reads=1000, annealing_time=100, debugging=False, is_cluster=True, shots=1)

solution = results.first.sample
print("Dynex Solution:", solution)

# Decode the selected stocks from the binary solution
weights = np.zeros(num_stocks)
for idx, val in solution.items():
    stock_index = int(idx[idx.find('[')+1 : idx.find(']')])
    if val == 1:
        weights[stock_index] = 1

# Portfolio statistics
portfolio_return = np.dot(weights, mean_returns.values)
portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
sharpe = portfolio_return / portfolio_risk

print("\nPortfolio Metrics:")
print(f"Expected Return: {portfolio_return:.4f} ({portfolio_return*100:.2f}%)")
print(f"Risk (Std Dev): {portfolio_risk:.4f} ({portfolio_risk*100:.2f}%)")
print(f"Sharpe Ratio: {sharpe:.4f}")

# Display portfolio weights
portfolio_series = pd.Series(weights, index=stocks)
selected = portfolio_series[portfolio_series > 0]

# Bar chart of selected weights
plt.figure(figsize=(10, 6))
selected.plot(kind='bar', color='skyblue')
plt.title('Portfolio Allocation')
plt.xlabel('Selected Stocks')
plt.ylabel('Weight')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Risk-return scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(np.sqrt(np.diag(cov_matrix.values)), mean_returns.values, s=100, label='Individual Stocks')

for i, ticker in enumerate(stocks):
    plt.annotate(ticker, (np.sqrt(cov_matrix.values[i, i]), mean_returns.values[i]), xytext=(5, 5), textcoords='offset points')

plt.scatter(portfolio_risk, portfolio_return, s=300, color='red', marker='*', label='Optimized Portfolio')
plt.grid(True)
plt.xlabel('Risk (Std Dev)')
plt.ylabel('Return')
plt.title('Risk vs Return')
plt.legend()
plt.tight_layout()
plt.show()

# Track the portfolio value over time
chosen_tickers = selected.index.tolist()
normalized_weights = selected / selected.sum()
portfolio_value = price_data[chosen_tickers].dot(normalized_weights)

plt.figure(figsize=(12, 6))
portfolio_value.plot(label='Optimized Portfolio', color='darkgreen')
plt.title('Portfolio Performance Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (Starting at 1)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
