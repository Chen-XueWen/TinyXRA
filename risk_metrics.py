import numpy as np
from scipy.stats import skew, kurtosis

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

def sortino_ratio(avg_abnormal_return, daily_residuals, mar=0.0):
    """
    Calculate the Sortino Ratio for abnormal returns.
    
    Parameters:
    - avg_abnormal_return: The average abnormal return (e.g., Jensen's Alpha).
    - daily_residuals: A pandas Series or NumPy array of daily abnormal returns (residuals).
    - mar: Minimum Acceptable Return for the abnormal return (default is 0.0).
    
    Returns:
    - Sortino Ratio.
    """
    # The "return" is the average abnormal return
    excess_return = avg_abnormal_return - mar
    
    # The "risk" is the downside deviation of the daily residuals
    downside_dev = downside_deviation(daily_residuals, mar)
    
    # Handle division by zero if there's no downside deviation
    if downside_dev == 0:
        return np.nan
        
    # Compute the Sortino Ratio
    sortino = excess_return / downside_dev
    return sortino

def calculate_standard_deviation(returns):
    return np.sqrt(np.mean(returns**2))

def calculate_skewness(returns):
    """Calculate skewness of returns"""
    return skew(returns, bias=False)

def calculate_kurtosis(returns):
    """Calculate excess kurtosis of returns"""
    return kurtosis(returns, fisher=True, bias=False)  # Fisher=True returns excess kurtosis (normal dist. = 0)
