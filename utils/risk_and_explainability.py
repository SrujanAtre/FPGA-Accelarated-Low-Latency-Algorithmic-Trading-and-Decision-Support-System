# =============================================================================
#  utils/risk_and_explainability.py  —  Risk management & decision explanations
#
#  Capstone innovation:
#  - Confidence-aware position sizing (Kelly Criterion / Fixed Fraction)
#  - Market regime adaptation
#  - Feature importance ranking
#  - Trade audit trail with reasoning
# =============================================================================

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """Risk classification."""
    CONSERVATIVE = "Conservative"
    MODERATE = "Moderate"
    AGGRESSIVE = "Aggressive"


@dataclass
class RiskAdjustedDecision:
    """Decision with risk parameters."""
    action: str                    # "BUY", "SELL", "HOLD"
    position_size_fraction: float  # Fraction of capital to risk
    recommended_stop_loss: float   # Stop loss price
    take_profit: float             # Take profit target
    risk_level: RiskLevel          # Risk classification
    Kelly_fraction: float          # Kelly % (0-1)
    expected_value: float          # Expected P&L


class RiskManager:
    """
    Manage position sizing and risk parameters based on model confidence
    and market conditions.
    
    Strategies:
    - Kelly Criterion: f* = (p * b - q) / b
      where p = win prob, q = lose prob, b = win/loss ratio
    - Fixed Fraction: risk fixed % of capital
    - Confidence-based: scale position by decision confidence
    """
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 kelly_fraction: float = 0.25,  # Use 25% of Kelly for safety
                 max_risk_per_trade: float = 0.02,  # Max 2% risk per trade
                 stop_loss_pct: float = 0.02,  # Stop at 2% loss
                 take_profit_pct: float = 0.05):  # Take profit at 5% gain
        """
        Initialize risk manager.
        
        Args:
            initial_capital: Portfolio size
            kelly_fraction: Fraction of full Kelly to use (0.25 = 25%)
            max_risk_per_trade: Max capital at risk per trade
            stop_loss_pct: Stop loss as % of entry price
            take_profit_pct: Take profit target as % of entry price
        """
        self.initial_capital = initial_capital
        self.kelly_fraction = kelly_fraction
        self.max_risk_per_trade = max_risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Win/loss tracking for adaptive sizing
        self.win_rate = 0.5  # Initial assumption
        self.avg_win = 0.01
        self.avg_loss = 0.01
    
    def size_position(self,
                      current_price: float,
                      decision_confidence: float,
                      market_regime: str = "stable",
                      historical_metrics: Dict = None) -> RiskAdjustedDecision:
        """
        Determine position size and risk parameters.
        
        Args:
            current_price: Entry price
            decision_confidence: Model confidence [0, 1]
            market_regime: "trending", "mean_revert", "volatile", "stable"
            historical_metrics: Win rate, avg win/loss from backtest
        
        Returns:
            RiskAdjustedDecision with sizing and stops
        """
        # Update with historical metrics if provided
        if historical_metrics:
            self.win_rate = historical_metrics.get('win_rate', 0.5)
            self.avg_win = historical_metrics.get('avg_win', 0.01)
            self.avg_loss = historical_metrics.get('avg_loss', 0.01)
        
        # Base position size from Kelly Criterion
        kelly_pct = self._compute_kelly(self.win_rate, self.avg_win, self.avg_loss)
        kelly_position = kelly_pct * self.kelly_fraction / self.initial_capital
        
        # Adjust for confidence
        confidence_adjusted = kelly_position * decision_confidence
        
        # Adjust for market regime
        regime_multiplier = {
            "trending": 1.2,
            "mean_revert": 0.9,
            "volatile": 0.7,
            "stable": 1.0
        }
        regime_adjusted = confidence_adjusted * regime_multiplier.get(market_regime, 1.0)
        
        # Cap at maximum risk
        final_position = min(regime_adjusted, self.max_risk_per_trade)
        
        # Determine risk level
        if final_position > 0.03:
            risk_level = RiskLevel.AGGRESSIVE
        elif final_position > 0.01:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.CONSERVATIVE
        
        # Calculate stops and targets
        stop_loss = current_price * (1 - self.stop_loss_pct)
        take_profit = current_price * (1 + self.take_profit_pct)
        
        # Expected value (rough estimate)
        expected_value = (self.win_rate * self.avg_win - (1 - self.win_rate) * self.avg_loss) * final_position * self.initial_capital
        
        return RiskAdjustedDecision(
            action="BUY",  # Placeholder; set by caller
            position_size_fraction=final_position,
            recommended_stop_loss=stop_loss,
            take_profit=take_profit,
            risk_level=risk_level,
            Kelly_fraction=kelly_pct,
            expected_value=expected_value
        )
    
    def _compute_kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Compute Kelly Criterion percentage.
        
        f* = (p * b - q) / b
        where p = win_rate, b = avg_win / avg_loss, q = 1 - p
        """
        if avg_loss == 0 or avg_win == 0:
            return 0.0
        
        b = avg_win / avg_loss
        q = 1 - win_rate
        
        kelly = (win_rate * b - q) / b
        
        # Clip to [0, 1] (can't risk more than 100% of capital)
        return np.clip(kelly, 0.0, 1.0)
    
    def update_metrics(self, win_rate: float, avg_win: float, avg_loss: float):
        """Update risk metrics from live trading or backtest."""
        self.win_rate = np.clip(win_rate, 0.0, 1.0)
        self.avg_win = max(avg_win, 0.0001)
        self.avg_loss = max(avg_loss, 0.0001)


class DecisionExplainer:
    """
    Explain trading decisions with feature importance and reasoning.
    
    Provides:
    - Top-N features driving the decision
    - Confidence decomposition
    - Risk assessment
    - Audit trail for compliance
    """
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.decision_log = []
    
    def explain(self,
                decision_action: str,
                decision_confidence: float,
                model_votes: Dict[str, str],
                feature_importance: List[Tuple[str, float]],
                current_price: float,
                market_regime: str = None) -> Dict:
        """
        Generate human-readable explanation for trading decision.
        
        Args:
            decision_action: "BUY", "SELL", or "HOLD"
            decision_confidence: Confidence score [0, 1]
            model_votes: Individual model decisions
            feature_importance: [(feature_name, importance_score), ...]
            current_price: Current market price
            market_regime: Current market regime
        
        Returns:
            Dictionary with explanation fields
        """
        explanation = {
            'action': decision_action,
            'confidence': float(decision_confidence),
            'confidence_category': self._categorize_confidence(decision_confidence),
            'primary_reason': self._generate_primary_reason(feature_importance),
            'supporting_features': feature_importance[:3],
            'model_agreement': self._assess_model_agreement(model_votes),
            'market_context': market_regime or "unknown",
            'risk_assessment': self._assess_risk(decision_confidence, feature_importance),
            'timestamp': str(np.datetime64('now'))
        }
        
        # Log for audit trail
        self.decision_log.append(explanation)
        
        return explanation
    
    def _categorize_confidence(self, confidence: float) -> str:
        """Categorize confidence level."""
        if confidence > 0.75:
            return "High"
        elif confidence > 0.55:
            return "Medium"
        else:
            return "Low"
    
    def _generate_primary_reason(self, feature_importance: List[Tuple]) -> str:
        """Generate text explanation of primary driver."""
        if not feature_importance:
            return "No dominant feature identified."
        
        top_feature, top_importance = feature_importance[0]
        
        reasons = {
            'rsi': f"RSI signal indicates {('overbought' if top_importance > 0.5 else 'oversold')} condition.",
            'macd': "MACD crossover signal detected.",
            'bb_width': f"Bollinger Band volatility {'expansion' if top_importance > 0.5 else 'contraction'} observed.",
            'volume': "Volume surge indicates institutional activity.",
            'ofi_proxy': "Order flow imbalance favoring this direction.",
            'vwap': "Price deviation from VWAP suggests momentum.",
            'atr': "Volatility is conducive to directional moves.",
            'close_norm': "Price action at key levels."
        }
        
        return reasons.get(top_feature, f"Feature '{top_feature}' is dominant.")
    
    def _assess_model_agreement(self, model_votes: Dict[str, str]) -> float:
        """Assess how much models agree (0-1, higher = more agreement)."""
        if not model_votes:
            return 0.0
        
        votes = list(model_votes.values())
        majority = max(set(votes), key=votes.count)
        agreement_count = votes.count(majority)
        
        return agreement_count / len(votes)
    
    def _assess_risk(self, confidence: float, feature_importance: List) -> str:
        """Assess risk of the decision."""
        if confidence > 0.75:
            return "Low risk (high confidence, aligned features)"
        elif confidence > 0.55 and len(feature_importance) > 2:
            return "Medium risk (moderate confidence, some disagreement)"
        else:
            return "High risk (low confidence, unclear signal) — consider skipping."
    
    def get_audit_trail(self, limit: int = None) -> List[Dict]:
        """Retrieve decision audit trail."""
        return self.decision_log[:limit] if limit else self.decision_log
    
    def print_explanation(self, explanation: Dict):
        """Pretty-print explanation."""
        print("\n" + "=" * 70)
        print(f"DECISION: {explanation['action']} (Confidence: {explanation['confidence']:.2%})")
        print("=" * 70)
        print(f"  Confidence Level:    {explanation['confidence_category']}")
        print(f"  Primary Reason:      {explanation['primary_reason']}")
        print(f"  Model Agreement:     {explanation['model_agreement']:.1%}")
        print(f"  Market Context:      {explanation['market_context']}")
        print(f"  Risk Assessment:     {explanation['risk_assessment']}")
        
        if explanation['supporting_features']:
            print(f"\n  Top Supporting Features:")
            for feat, importance in explanation['supporting_features']:
                print(f"    • {feat:<20} {importance:>8.4f}")
        
        print("=" * 70 + "\n")


if __name__ == "__main__":
    print("Risk & Explainability Module — Example Usage\n")
    
    # Risk Manager Example
    risk_mgr = RiskManager(initial_capital=100000)
    risk_mgr.update_metrics(win_rate=0.55, avg_win=0.02, avg_loss=0.01)
    
    decision = risk_mgr.size_position(
        current_price=100.0,
        decision_confidence=0.75,
        market_regime="trending"
    )
    print(f"Position Size: {decision.position_size_fraction:.4f} ({decision.position_size_fraction*100:.2f}% of capital)")
    print(f"Stop Loss: {decision.recommended_stop_loss:.2f}")
    print(f"Take Profit: {decision.take_profit:.2f}")
    print(f"Risk Level: {decision.risk_level.value}")
    
    # Decision Explainer Example
    explainer = DecisionExplainer(['RSI', 'MACD', 'BB_Width', 'Volume', 'OFI_Proxy'])
    
    explanation = explainer.explain(
        decision_action="BUY",
        decision_confidence=0.78,
        model_votes={'lstm': 'BUY', 'cnn': 'BUY', 'lgbm': 'HOLD', 'rf': 'BUY'},
        feature_importance=[('RSI', 0.35), ('MACD', 0.25), ('Volume', 0.20)],
        current_price=100.0,
        market_regime="trending"
    )
    
    explainer.print_explanation(explanation)
