"""
Enhanced Neural Network Strategy for Hyperliquid Master
Integrated from Ultimate Master Bot
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, Optional, List, Tuple
import time
import json

# Technical Analysis
import ta
from ta.trend import macd, macd_signal, ADXIndicator
from ta.momentum import rsi, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

from utils.logger import get_logger
from strategies.base_strategy import BaseStrategy
from strategies.trading_types_fixed import TradingSignal, SignalType, MarketData

logger = get_logger(__name__)

class TransformerPriceModel(nn.Module):
    """
    Advanced Transformer-based price prediction model
    Uses 12 features per bar for comprehensive market analysis
    """
    
    def __init__(self, input_size_per_bar=12, lookback_bars=30, hidden_size=64, dropout_p=0.1):
        super().__init__()
        self.lookback_bars = lookback_bars
        self.input_size_per_bar = input_size_per_bar
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = nn.Linear(input_size_per_bar, hidden_size)
        
        # Transformer architecture
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=hidden_size,
            dropout=dropout_p,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout_p)
        
        # Output heads
        self.reg_head = nn.Linear(hidden_size, 1)  # Price prediction
        self.cls_head = nn.Linear(hidden_size, 3)  # Direction classification (buy/sell/hold)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through the transformer model
        
        Args:
            x: Input tensor of shape (batch_size, lookback_bars * input_size_per_bar)
            
        Returns:
            Tuple of (price_prediction, direction_classification)
        """
        bsz = x.shape[0]
        x = x.view(bsz, self.lookback_bars, self.input_size_per_bar)
        
        # Embedding
        x = self.embedding(x)
        
        # Transformer
        out = self.transformer(x, x)
        
        # Use last output
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        
        # Predictions
        reg = self.reg_head(last_out)  # Price prediction
        cls = self.cls_head(last_out)  # Direction classification
        
        return reg, cls


class RLParameterTuner:
    """
    Reinforcement Learning Parameter Tuner
    Automatically optimizes trading parameters based on performance
    """
    
    def __init__(self, config: dict, param_state_path: str):
        self.config = config
        self.param_state_path = param_state_path
        self.episode_pnl = 0.0
        self.trade_count = 0
        self.best_pnl = -float("inf")
        self.best_params = {}
        self.losing_streak = 0
        self.winning_streak = 0
        self.original_order_size = self.config.get("order_size", 0.1)
        self.cooldown_until = 0
        self.load_state()
    
    def load_state(self):
        """Load saved parameter tuning state"""
        if os.path.exists(self.param_state_path):
            try:
                with open(self.param_state_path, "r") as f:
                    state = json.load(f)
                self.best_pnl = state.get("best_pnl", -float("inf"))
                self.best_params = state.get("best_params", {})
                logger.info(f"Loaded parameter tuner state: best_pnl={self.best_pnl}")
            except Exception as e:
                logger.error(f"Failed to load parameter state: {e}")
        else:
            logger.info("No existing parameter state found; starting fresh")
    
    def save_state(self):
        """Save parameter tuning state"""
        try:
            state = {
                "best_pnl": self.best_pnl,
                "best_params": self.best_params
            }
            with open(self.param_state_path, "w") as f:
                json.dump(state, f, indent=2)
            logger.info(f"Saved parameter tuner state")
        except Exception as e:
            logger.error(f"Failed to save parameter state: {e}")
    
    def on_trade_closed(self, trade_pnl: float):
        """
        Process completed trade and adjust parameters
        
        Args:
            trade_pnl: Profit/loss from the completed trade
        """
        self.episode_pnl += trade_pnl
        self.trade_count += 1
        
        # Track streaks
        if trade_pnl < 0:
            self.losing_streak += 1
            self.winning_streak = 0
        else:
            self.losing_streak = 0
            self.winning_streak += 1
        
        # Evaluate parameters every 5 trades
        if self.trade_count % 5 == 0:
            self.evaluate_params()
        
        # Adjust order size based on streaks
        if self.losing_streak >= 3:
            old_size = self.config.get("order_size", 0.1)
            new_size = old_size * 0.75
            self.config["order_size"] = new_size
            self.cooldown_until = time.time() + 30
            logger.info(f"Losing streak detected: reducing order size {old_size} -> {new_size}")
            self.losing_streak = 0
        
        if self.winning_streak >= 3:
            old_size = self.config.get("order_size", 0.1)
            new_size = min(old_size * 1.15, 2.0 * self.original_order_size)
            self.config["order_size"] = new_size
            logger.info(f"Winning streak detected: increasing order size {old_size} -> {new_size}")
            self.winning_streak = 0
    
    def evaluate_params(self):
        """Evaluate current parameters and optimize if needed"""
        if self.episode_pnl > self.best_pnl:
            self.best_pnl = self.episode_pnl
            self.best_params = {
                "stop_loss_pct": self.config.get("stop_loss_pct", 0.02),
                "take_profit_pct": self.config.get("take_profit_pct", 0.04),
                "order_size": self.config.get("order_size", 0.1)
            }
            self.save_state()
            logger.info(f"New best parameters found with PnL: {self.best_pnl}")
        else:
            self.random_nudge()
        
        self.episode_pnl = 0.0
    
    def random_nudge(self):
        """Randomly adjust parameters to explore better configurations"""
        import random
        
        params = ["stop_loss_pct", "take_profit_pct", "order_size"]
        chosen = random.choice(params)
        old_value = self.config.get(chosen, 0.1)
        
        if chosen == "stop_loss_pct":
            new_value = max(0.005, min(0.1, old_value + random.uniform(-0.005, 0.005)))
        elif chosen == "take_profit_pct":
            new_value = max(0.01, min(0.2, old_value + random.uniform(-0.01, 0.01)))
        elif chosen == "order_size":
            new_value = max(0.01, min(1.0, old_value + random.uniform(-0.05, 0.05)))
        else:
            new_value = old_value
        
        self.config[chosen] = new_value
        logger.info(f"Parameter nudge: {chosen} {old_value} -> {new_value}")
    
    def is_in_cooldown(self) -> bool:
        """Check if tuner is in cooldown period"""
        return time.time() < self.cooldown_until


class EnhancedNeuralStrategy(BaseStrategy):
    """
    Enhanced Neural Network Strategy with Transformer model and RL parameter tuning
    """
    
    def __init__(self, api=None, risk_manager=None, max_positions=3):
        super().__init__(api, risk_manager, max_positions)
        
        # Strategy parameters
        self.lookback_bars = 30
        self.features_per_bar = 12
        self.hidden_size = 64
        self.learning_rate = 0.0003
        self.confidence_threshold = 0.7
        
        # Technical indicator parameters
        self.fast_ma_period = 5
        self.slow_ma_period = 15
        self.rsi_period = 14
        self.bb_period = 20
        self.bb_std = 2.0
        
        # Risk management
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.04
        self.trailing_stop = True
        self.partial_tp_levels = [0.02, 0.04]
        self.partial_tp_ratios = [0.3, 0.3]
        
        # Initialize components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerPriceModel(
            input_size_per_bar=self.features_per_bar,
            lookback_bars=self.lookback_bars,
            hidden_size=self.hidden_size
        ).to(self.device)
        
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.95)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Data storage
        self.historical_data = pd.DataFrame()
        self.training_data = []
        
        # Parameter tuner
        self.param_tuner = RLParameterTuner(
            config={
                "stop_loss_pct": self.stop_loss_pct,
                "take_profit_pct": self.take_profit_pct,
                "order_size": 0.1
            },
            param_state_path="enhanced_neural_params.json"
        )
        
        # Model persistence
        self.model_path = "enhanced_neural_model.pth"
        self.load_model()
        
        logger.info("Enhanced Neural Strategy initialized with Transformer model and RL tuning")
    
    def load_model(self):
        """Load saved model if available"""
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
        else:
            logger.info("No saved model found, starting with fresh model")
    
    def save_model(self):
        """Save current model"""
        try:
            torch.save(self.model.state_dict(), self.model_path)
            logger.info(f"Saved model to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        if len(df) < max(self.slow_ma_period, self.rsi_period, self.bb_period):
            return df
        
        try:
            # Moving averages
            df['fast_ma'] = df['close'].rolling(window=self.fast_ma_period).mean()
            df['slow_ma'] = df['close'].rolling(window=self.slow_ma_period).mean()
            
            # RSI
            df['rsi'] = rsi(df['close'], window=self.rsi_period)
            
            # MACD
            df['macd'] = macd(df['close'])
            df['macd_signal'] = macd_signal(df['close'])
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            bb = BollingerBands(df['close'], window=self.bb_period, window_dev=self.bb_std)
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_mid'] = bb.bollinger_mavg()
            
            # Stochastic
            stoch = StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # ADX
            adx = ADXIndicator(df['high'], df['low'], df['close'])
            df['adx'] = adx.adx()
            
            # ATR
            atr = AverageTrueRange(df['high'], df['low'], df['close'])
            df['atr'] = atr.average_true_range()
            
            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(window=10).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix for the neural network
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Feature matrix with shape (samples, lookback_bars * features_per_bar)
        """
        if len(df) < self.lookback_bars:
            return np.array([])
        
        # Select features (12 features per bar)
        feature_columns = [
            'close', 'volume', 'fast_ma', 'slow_ma', 'rsi', 'macd_hist',
            'bb_high', 'bb_low', 'stoch_k', 'stoch_d', 'adx', 'atr'
        ]
        
        # Ensure all columns exist
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        
        # Extract features
        features = df[feature_columns].values
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create sliding windows
        samples = []
        for i in range(self.lookback_bars, len(features)):
            window = features[i-self.lookback_bars:i]
            samples.append(window.flatten())
        
        if not samples:
            return np.array([])
        
        features_array = np.array(samples)
        
        # Scale features
        if len(features_array) > 0:
            features_array = self.scaler.fit_transform(features_array)
        
        return features_array
    
    def predict_signal(self, market_data: MarketData) -> TradingSignal:
        """
        Generate trading signal using the neural network
        
        Args:
            market_data: Current market data
            
        Returns:
            Trading signal
        """
        try:
            # Update historical data
            new_row = {
                'timestamp': market_data.timestamp,
                'open': market_data.open,
                'high': market_data.high,
                'low': market_data.low,
                'close': market_data.close,
                'volume': market_data.volume
            }
            
            self.historical_data = pd.concat([
                self.historical_data,
                pd.DataFrame([new_row])
            ], ignore_index=True)
            
            # Keep only recent data
            if len(self.historical_data) > 1000:
                self.historical_data = self.historical_data.tail(500)
            
            # Calculate technical indicators
            df_with_indicators = self.calculate_technical_indicators(self.historical_data.copy())
            
            # Prepare features
            features = self.prepare_features(df_with_indicators)
            
            if len(features) == 0:
                return TradingSignal(
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    entry_price=market_data.close,
                    stop_loss=None,
                    take_profit=None,
                    metadata={"reason": "insufficient_data"}
                )
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                # Use the last sample for prediction
                input_tensor = torch.FloatTensor(features[-1:]).to(self.device)
                price_pred, direction_pred = self.model(input_tensor)
                
                # Get direction probabilities
                direction_probs = F.softmax(direction_pred, dim=1)
                direction_class = torch.argmax(direction_probs, dim=1).item()
                confidence = torch.max(direction_probs).item()
            
            # Generate signal based on prediction
            if confidence < self.confidence_threshold:
                signal_type = SignalType.HOLD
            elif direction_class == 0:  # Sell
                signal_type = SignalType.SELL
            elif direction_class == 2:  # Buy
                signal_type = SignalType.BUY
            else:  # Hold
                signal_type = SignalType.HOLD
            
            # Calculate stop loss and take profit
            current_price = market_data.close
            stop_loss = None
            take_profit = None
            
            if signal_type == SignalType.BUY:
                stop_loss = current_price * (1 - self.stop_loss_pct)
                take_profit = current_price * (1 + self.take_profit_pct)
            elif signal_type == SignalType.SELL:
                stop_loss = current_price * (1 + self.stop_loss_pct)
                take_profit = current_price * (1 - self.take_profit_pct)
            
            return TradingSignal(
                signal_type=signal_type,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    "model_prediction": price_pred.item() if price_pred is not None else 0.0,
                    "direction_class": direction_class,
                    "neural_confidence": confidence
                }
            )
            
        except Exception as e:
            logger.error(f"Error in neural network prediction: {e}")
            return TradingSignal(
                signal_type=SignalType.HOLD,
                confidence=0.0,
                entry_price=market_data.close,
                stop_loss=None,
                take_profit=None,
                metadata={"error": str(e)}
            )
    
    def on_trade_closed(self, entry_price: float, exit_price: float, quantity: float, side: str):
        """
        Handle trade closure for learning and parameter tuning
        
        Args:
            entry_price: Entry price of the trade
            exit_price: Exit price of the trade
            quantity: Trade quantity
            side: Trade side ('buy' or 'sell')
        """
        try:
            # Calculate PnL
            if side.lower() == 'buy':
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity
            
            # Update parameter tuner
            self.param_tuner.on_trade_closed(pnl)
            
            # Update strategy parameters from tuner
            self.stop_loss_pct = self.param_tuner.config.get("stop_loss_pct", self.stop_loss_pct)
            self.take_profit_pct = self.param_tuner.config.get("take_profit_pct", self.take_profit_pct)
            
            logger.info(f"Trade closed: PnL={pnl:.4f}, Updated parameters: SL={self.stop_loss_pct:.4f}, TP={self.take_profit_pct:.4f}")
            
        except Exception as e:
            logger.error(f"Error handling trade closure: {e}")
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default strategy parameters"""
        return {
            'lookback_bars': self.lookback_bars,
            'confidence_threshold': self.confidence_threshold,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'fast_ma_period': self.fast_ma_period,
            'slow_ma_period': self.slow_ma_period,
            'rsi_period': self.rsi_period,
            'bb_period': self.bb_period,
            'bb_std': self.bb_std
        }

