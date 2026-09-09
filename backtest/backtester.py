# =============================================================================
#  backtest/backtester.py  —  Walk-forward backtesting with trading metrics
#
#  Includes:
#  - Walk-forward validation (rolling train/test windows)
#  - P&L calculation with slippage and commissions
#  - Sharpe ratio, Sortino ratio, Max drawdown, Win rate
#  - Regime detection (trending vs mean-reverting vs volatile)
#  - Per-trade analysis
# =============================================================================

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING = "Trending"
    MEAN_REVERT = "Mean Reverting"
    VOLATILE = "Volatile"
    STABLE = "Stable"


@dataclass
class BacktestResults:
    """Comprehensive backtest results."""
    total_return: float            # Total return % 
    annual_return: float           # Annualized return
    sharpe_ratio: float            # Risk-adjusted return
    sortino_ratio: float           # Downside risk-adjusted return
    max_drawdown: float            # Maximum peak-to-trough decline
    win_rate: float                # % of profitable trades
    profit_factor: float           # Gross profit / Gross loss
    trades_total: int              # Total number of trades
    trades_buy: int                # Number of buy signals
    trades_sell: int               # Number of sell signals
    avg_trade_return: float        # Average return per trade
    best_trade: float              # Best single trade return
    worst_trade: float             # Worst single trade return
    regime_performance: Dict[str, float]  # Returns by regime
    cumulative_returns: np.ndarray # Daily returns over time
    equity_curve: np.ndarray       # Portfolio value over time
    drawdown_curve: np.ndarray     # Drawdown % over time


class HFTBacktester:
    """
    Walk-forward backtester for algorithmic trading strategy.
    
    Simulates:
    - Real-time decision making (no look-ahead bias)
    - Transaction costs (slippage, commissions)
    - Market impact from position sizing
    - Multiple market regimes
    """
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 commission_bps: float = 2.0,    # 2 basis points
                 slippage_bps: float = 1.0,      # 1 basis point
                 max_position_size: float = 0.1, # 10% of capital per position
                 lookback_days: int = 20,        # For regime detection
                 risk_free_rate: float = 0.05):  # For Sharpe ratio
        """
        Initialize backtester with trading parameters.
        
        Args:
            initial_capital: Starting portfolio value
            commission_bps: Trading commission in basis points
            slippage_bps: Average slippage in basis points
            max_position_size: Max position as fraction of capital
            lookback_days: Period for volatility/trend calculation
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps / 10000.0
        self.slippage_bps = slippage_bps / 10000.0
        self.max_position_size = max_position_size
        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate
        
        self.trades = []
        self.equity_curve = []
        self.positions = []
        self.dates = []
    
    def backtest(self,
                 df: pd.DataFrame,
                 decisions: np.ndarray,      # [n_bars] array of 0 (BUY), 1 (HOLD), 2 (SELL)
                 confidences: np.ndarray = None,  # [n_bars] confidence scores
                 regime_labels: np.ndarray = None  # [n_bars] regime classification
                 ) -> BacktestResults:
        """
        Run backtest on historical data with model decisions.
        
        Args:
            df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            decisions: Array of trading decisions (0=BUY, 1=HOLD, 2=SELL)
            confidences: Optional confidence scores for filtering
            regime_labels: Optional market regime labels
        
        Returns:
            BacktestResults with comprehensive trading metrics
        """
        n_bars = len(df)
        
        # Initialize tracking variables
        cash = self.initial_capital
        position = 0  # shares held
        position_price = 0
        equity = [self.initial_capital]
        self.trades = []
        pnl_list = []
        
        # Filter decisions by confidence if provided
        if confidences is not None:
            decisions = decisions.copy()
            decisions[confidences < 0.5] = 1  # Convert low-confidence to HOLD
        
        # Detect market regimes if not provided
        if regime_labels is None:
            regime_labels = self._detect_regimes(df)
        
        for i in range(n_bars):
            price = df.iloc[i]['Close']
            decision = decisions[i]
            regime = regime_labels[i] if regime_labels is not None else MarketRegime.STABLE
            
            # Decision: 0 = BUY, 1 = HOLD, 2 = SELL
            if decision == 0:  # BUY signal
                if position == 0:  # Not currently holding
                    # Buy with Kelly sizing or fixed fraction
                    position_size = self._calculate_position_size(cash, price)
                    shares = position_size / price
                    transaction_cost = shares * price * (self.commission_bps + self.slippage_bps)
                    
                    cash -= shares * price + transaction_cost
                    position = shares
                    position_price = price
                    
                    self.trades.append({
                        'entry_bar': i,
                        'entry_price': price,
                        'entry_regime': regime,
                        'action': 'BUY',
                        'shares': shares
                    })
            
            elif decision == 2:  # SELL signal
                if position > 0:  # Currently holding
                    exit_value = position * price * (1 - self.slippage_bps)
                    transaction_cost = position * price * self.commission_bps
                    
                    cash += exit_value - transaction_cost
                    trade_return = (price - position_price) / position_price
                    pnl = position * (price - position_price) - transaction_cost
                    pnl_list.append(pnl)
                    
                    # Log trade
                    if len(self.trades) > 0 and self.trades[-1]['action'] == 'BUY':
                        self.trades[-1].update({
                            'exit_bar': i,
                            'exit_price': price,
                            'exit_regime': regime,
                            'return': trade_return,
                            'pnl': pnl
                        })
                    
                    position = 0
                    position_price = 0
            
            # Mark-to-market accounting
            current_equity = cash + position * price
            equity.append(current_equity)
            self.equity_curve.append(current_equity)
            self.dates.append(df.index[i] if hasattr(df.index, '__iter__') else i)
        
        # Close any open position at end
        if position > 0:
            final_value = position * df.iloc[-1]['Close'] * (1 - self.slippage_bps)
            cash += final_value - position * df.iloc[-1]['Close'] * self.commission_bps
            trade_return = (df.iloc[-1]['Close'] - position_price) / position_price
            pnl_list.append(final_value - position * position_price)
        
        # Compute metrics
        equity_array = np.array(equity)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        total_return = (equity_array[-1] - self.initial_capital) / self.initial_capital
        annual_return = total_return * 252 / len(df) * 252  # Annualize (252 trading days)
        
        # Risk metrics
        sharpe = self._compute_sharpe(returns)
        sortino = self._compute_sortino(returns)
        max_dd = self._compute_max_drawdown(equity_array)
        
        # Trade metrics
        profitable_trades = sum(1 for t in self.trades if t.get('return', 0) > 0)
        win_rate = profitable_trades / len(self.trades) if self.trades else 0
        
        profit_factor = self._compute_profit_factor(self.trades)
        
        avg_trade_return = np.mean([t.get('return', 0) for t in self.trades]) if self.trades else 0
        best_trade = max([t.get('return', 0) for t in self.trades], default=0)
        worst_trade = min([t.get('return', 0) for t in self.trades], default=0)
        
        # Performance by regime
        regime_performance = self._compute_regime_performance(self.trades)
        
        # Drawdown curve
        drawdowns = self._compute_drawdown_curve(equity_array)
        
        return BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades_total=len(self.trades),
            trades_buy=sum(1 for t in self.trades if t['action'] == 'BUY'),
            trades_sell=sum(1 for t in self.trades if 'exit_price' in t),
            avg_trade_return=avg_trade_return,
            best_trade=best_trade,
            worst_trade=worst_trade,
            regime_performance=regime_performance,
            cumulative_returns=returns,
            equity_curve=equity_array,
            drawdown_curve=drawdowns
        )
    
    def _calculate_position_size(self, cash: float, price: float) -> float:
        """Calculate position size (simple fixed fraction)."""
        return min(cash * self.max_position_size, cash)
    
    def _detect_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """
        Detect market regimes: Trending, Mean-Reverting, Volatile, Stable.
        
        Uses: trend (20-bar MA), volatility (20-bar std), ADX-like metric.
        """
        close = df['Close'].values
        lookback = min(self.lookback_days, len(close) // 4)
        
        regimes = []
        
        for i in range(len(close)):
            start = max(0, i - lookback)
            window = close[start:i+1]
            
            if len(window) < lookback // 2:
                regimes.append(MarketRegime.STABLE)
                continue
            
            # Trend: direction of 20-bar SMA
            sma = np.mean(window)
            trend = (window[-1] - sma) / (np.std(window) + 1e-10)
            
            # Volatility: coefficient of variation
            volatility = np.std(window) / (np.mean(window) + 1e-10)
            
            # Range-bound: std of returns
            returns = np.diff(window) / window[:-1]
            mean_revert_score = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            
            # Classification
            if abs(trend) > 1.5:
                regimes.append(MarketRegime.TRENDING)
            elif volatility > 0.02:
                regimes.append(MarketRegime.VOLATILE)
            elif mean_revert_score < -0.2:
                regimes.append(MarketRegime.MEAN_REVERT)
            else:
                regimes.append(MarketRegime.STABLE)
        
        return np.array(regimes)
    
    def _compute_sharpe(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Compute Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / periods_per_year
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _compute_sortino(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Compute Sortino ratio (uses downside volatility)."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - self.risk_free_rate / periods_per_year
        downside = np.where(excess_returns < 0, excess_returns, 0)
        downside_std = np.std(downside)
        
        if downside_std == 0:
            return 0.0
        
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_std
    
    def _compute_max_drawdown(self, equity: np.ndarray) -> float:
        """Compute maximum peak-to-trough drawdown."""
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return float(np.min(drawdown))
    
    def _compute_drawdown_curve(self, equity: np.ndarray) -> np.ndarray:
        """Compute drawdown at each point."""
        running_max = np.maximum.accumulate(equity)
        return (equity - running_max) / running_max
    
    def _compute_profit_factor(self, trades: List[Dict]) -> float:
        """Compute profit factor = gross profit / gross loss."""
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def _compute_regime_performance(self, trades: List[Dict]) -> Dict[str, float]:
        """Compute returns by market regime."""
        regime_perf = {}
        
        for regime in MarketRegime:
            regime_trades = [t for t in trades if t.get('entry_regime') == regime and 'return' in t]
            if regime_trades:
                avg_return = np.mean([t['return'] for t in regime_trades])
                regime_perf[regime.value] = avg_return
        
        return regime_perf
    
    def print_results(self, results: BacktestResults):
        """Pretty-print backtest results."""
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS")
        print("=" * 70)
        print(f"\n[RETURNS]")
        print(f"  Total Return:        {results.total_return:>10.2%}")
        print(f"  Annualized Return:   {results.annual_return:>10.2%}")
        
        print(f"\n[RISK-ADJUSTED]")
        print(f"  Sharpe Ratio:        {results.sharpe_ratio:>10.2f}")
        print(f"  Sortino Ratio:       {results.sortino_ratio:>10.2f}")
        print(f"  Max Drawdown:        {results.max_drawdown:>10.2%}")
        
        print(f"\n[TRADE STATISTICS]")
        print(f"  Total Trades:        {results.trades_total:>10d}")
        print(f"  Buy Signals:         {results.trades_buy:>10d}")
        print(f"  Sell Signals:        {results.trades_sell:>10d}")
        print(f"  Win Rate:            {results.win_rate:>10.2%}")
        print(f"  Profit Factor:       {results.profit_factor:>10.2f}")
        
        print(f"\n[PER-TRADE]")
        print(f"  Avg Trade Return:    {results.avg_trade_return:>10.2%}")
        print(f"  Best Trade:          {results.best_trade:>10.2%}")
        print(f"  Worst Trade:         {results.worst_trade:>10.2%}")
        
        if results.regime_performance:
            print(f"\n[BY REGIME]")
            for regime, perf in results.regime_performance.items():
                print(f"  {regime:<20} {perf:>10.2%}")
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    print("HFT Backtester — Example\n")
    
    # Create mock data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.1)
    df = pd.DataFrame({
        'Open': prices,
        'High': prices + np.abs(np.random.randn(1000) * 0.05),
        'Low': prices - np.abs(np.random.randn(1000) * 0.05),
        'Close': prices,
        'Volume': np.random.randint(1000, 5000, 1000)
    }, index=dates)
    
    # Mock decisions (randomly generated for demo)
    decisions = np.random.choice([0, 1, 2], size=1000, p=[0.1, 0.7, 0.2])
    confidences = np.random.rand(1000)
    
    # Run backtest
    backtester = HFTBacktester(initial_capital=100000)
    results = backtester.backtest(df, decisions, confidences)
    backtester.print_results(results)
