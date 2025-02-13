import numpy as np

def downside_deviation(returns, mar=0.0):
    """
    Calculate the downside deviation of returns.
    
    Parameters:
    - returns: A pandas Series or NumPy array of returns.
    - mar: Minimum Acceptable Return (default is 0.0).
    
    Returns:
    - Downside deviation.
    """
    # Calculate the differences between returns and MAR
    diff = returns - mar
    # Filter only the negative differences
    negative_diff = diff[diff < 0]
    # Square the negative differences
    squared_diff = negative_diff ** 2
    # Calculate the mean of the squared negative differences
    mean_squared_diff = np.mean(squared_diff)
    # Take the square root to obtain the downside deviation
    downside_dev = np.sqrt(mean_squared_diff)
    return downside_dev

def sortino_ratio(returns, mar=0.0, rf=0.0):
    """
    Calculate the Sortino Ratio of returns.
    
    Parameters:
    - returns: A pandas Series or NumPy array of returns.
    - mar: Minimum Acceptable Return (default is 0.0).
    - rf: Risk-free rate (default is 0.0). # We have adjusted as Excess returns, so we can take it as 0.0
    
    Returns:
    - Sortino Ratio.
    """
    # Calculate the average return above the risk-free rate
    excess_return = np.mean(returns) - rf
    # Calculate the downside deviation
    downside_dev = downside_deviation(returns, mar)
    # Compute the Sortino Ratio
    sortino = excess_return / downside_dev
    return sortino

def standard_deviation(returns):
    return np.sqrt(np.mean(returns**2))