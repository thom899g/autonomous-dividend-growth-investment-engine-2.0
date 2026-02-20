import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dividend_growth_investor.log'),
        logging.StreamHandler()
    ]
)

def initialize_logging() -> None:
    """
    Initializes the logging system with both file and console handlers.
    """
    pass  # Logging is already initialized above

class DividendGrowthInvestor:
    def __init__(self, portfolio_value: float = 10000.0) -> None:
        """
        Initialize the dividend growth investor engine.
        
        Args:
            portfolio_value: Initial portfolio value in dollars
        """
        self.portfolio_value = portfolio_value
        self.dividend_data: Dict[str, pd.DataFrame] = {}
        self.reinvestment_threshold = 0.05
        
    def fetch_dividend_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetches historical dividend data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            
        Returns:
            DataFrame containing dividend information or None if failed
        """
        try:
            # Get dividend data from Yahoo Finance
            df = yf.Ticker(ticker).info['dividendHistory']
            return pd.DataFrame(df)
        except Exception as e:
            logging.error(f"Failed to fetch dividend data for {ticker}: {str(e)}")
            return None
    
    def select_dividend_stocks(self, tickers: List[str]) -> Dict[str, float]:
        """
        Selects stocks from a list based on dividend growth criteria.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary mapping tickers to their predicted growth rates
        """
        selected = {}
        for ticker in tickers:
            data = self.fetch_dividend_data(ticker)
            if data is not None and not data.empty:
                # Calculate growth rate
                growth_rate = self.calculate_growth_rate(data['dividend'].values)
                if growth_rate > 0.05:  # Select only stocks with above 5% growth
                    selected[ticker] = growth_rate
        return selected
    
    def reinvest_dividends(self, allocations: Dict[str, float]) -> None:
        """
        Reinvests dividends into selected stocks based on optimal allocation.
        
        Args:
            allocations: Dictionary of stock allocations
        """
        total_dividends = sum(self.portfolio_value * (allocation / 100) for allocation in allocations.values())
        if total_dividends < 100:  # Only reinvest if we have enough to make meaningful investments
            return
            
        try:
            # Calculate optimal portfolio weights
            weights = self.calculate_optimal_weights(allocations)
            
            # Reinvest into selected stocks
            for ticker, weight in weights.items():
                investment_amount = (self.portfolio_value * weight) + (total_dividends * 0.25)
                self.add_to_portfolio(ticker, investment_amount)
            logging.info("Successfully reinvested dividends.")
        except Exception as e:
            logging.error(f"Failed to reinvest dividends: {str(e)}")
    
    @staticmethod
    def calculate_growth_rate(dividend_values: np.ndarray) -> float:
        """
        Calculates the compound annual growth rate (CAGR) of dividends.
        
        Args:
            dividend_values: Array of historical dividend values
            
        Returns:
            CAGR as a decimal (e.g., 0.05 for 5%)
        """
        if len(dividend_values) < 2:
            return 0.0
        n = len(dividend_values) - 1
        cagr = (dividend_values[-1] / dividend_values[0]) ** (1/n) - 1
        return max(cagr, 0)
    
    @staticmethod
    def calculate_dividend_yield(price: float, dividend: float) -> float:
        """
        Calculates the dividend yield ratio.
        
        Args:
            price: Stock price per share
            dividend: Annual dividend per share
            
        Returns:
            Dividend yield as a decimal (e.g., 0.05 for 5%)
        """
        return dividend / price if price != 0 else 0.0
    
    @staticmethod
    def get_historical_volatility(prices: pd.Series, window: int = 252) -> float:
        """
        Calculates historical volatility over a specified window.
        
        Args:
            prices: DataFrame column of closing prices
            window: Number of periods to consider
            
        Returns:
            Volatility as a decimal (e.g., 0.15 for 15%)
        """
        if len(prices) < window:
            return 0.0
        returns = np.log(prices.pct_change().dropna())
        volatility = returns.std() * np.sqrt(252)  # Annualize the volatility
        return max(volatility, 0.0)
    
    def calculate_optimal_weights(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """
        Determines optimal portfolio weights based on risk and return.
        
        Args:
            allocations: Dictionary of stock allocations
            
        Returns:
            Optimized weight distribution
        """
        # Simple mean-variance optimization for demonstration
        returns = [allocation for allocation in allocations.values()]
        covariance_matrix = np.cov(returns)
        inv_cov = np.linalg.inv(covariance_matrix)
        
        ones = np.array([1.0]*len(inv_cov)).T
        denominator = ones.T.dot(inv_cov).dot(ones)
        weights = inv_cov.dot(ones) / denominator
        
        return {ticker: weight for ticker, _ in zip(allocations.keys(), weights)}
    
    def add_to_portfolio(self, ticker: str, amount: float) -> None:
        """
        Adds a position to the portfolio.
        
        Args:
            ticker: Stock ticker
            amount: Investment amount
        """
        # In a real system, this would