import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('C:\Users\John\Documents\OCDO.L.csv')

# Calculate single VaR and ES using statistical approach
VaR_stat = data['Close'].quantile(0.05)
ES_stat = data['Close'].iloc[:int(0.05 * len(data))].mean()

# Calculate portfolio VaR and ES using variance-covariance approach
cov_matrix = data[['Close']].cov()
w = np.array([0.5, 0.5])
portfolio_return = data['Close'].dot(w)
portfolio_VaR_cov = np.sqrt(w.dot(cov_matrix).dot(w.T)) * 1.645
portfolio_ES_cov = portfolio_return - portfolio_VaR_cov

# Calculate single VaR and ES using historical simulation approach
VaR_hist = []
ES_hist = []
for i in range(len(data)):
    window = data['Close'].iloc[i - 250:i]
    VaR_hist.append(window.quantile(0.05))
    ES_hist.append(window.iloc[:int(0.05 * len(window))].mean())

# Calculate portfolio VaR and ES using historical simulation approach
portfolio_VaR_hist = []
portfolio_ES_hist = []
for i in range(len(data)):
    window = data[['Close']].iloc[i - 250:i]
    portfolio_return = window.dot(w)
    portfolio_VaR_hist.append(np.sqrt(w.dot(window.cov()).dot(w.T)) * 1.645)
    portfolio_ES_hist.append(portfolio_return - portfolio_VaR_hist[-1])

# Create a DataFrame to store the results
result = pd.DataFrame({
    'Date': data['Date'],
    'Close': data['Close'],
    'Portfolio Return': portfolio_return,
    'VaR (Statistical Approach)': VaR_stat,
    'ES (Statistical Approach)': ES_stat,
    'Portfolio VaR (Variance-Covariance)': portfolio_VaR_cov,
    'Portfolio ES (Variance-Covariance)': portfolio_ES_cov,
    'VaR (Historical Simulation)': VaR_hist,
    'ES (Historical Simulation)': ES_hist,
    'Portfolio VaR (Historical Simulation)': portfolio_VaR_hist,
    'Portfolio ES (Historical Simulation)': portfolio_ES_hist
})

# Save the results to a CSV file
result.to_csv('portfolio_results.csv', index=False)

# Print a message indicating that the file has been saved
print('Results saved to "portfolio_results.csv"')
