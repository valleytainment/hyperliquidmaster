#!/usr/bin/env python3
"""
Enhanced Neural Network Strategy with RL Parameter Tuning
---------------------------------------------------------
Features:
• Transformer-based neural network with 12 features per bar
• RL parameter optimization for automatic tuning
• Advanced technical indicators (MA, RSI, MACD, Bollinger Bands, ADX, ATR, Stochastic)
• Synergy confidence threshold for signal validation
• Robust error handling and asynchronous training
• Dynamic position sizing and risk management
"""

import os
import time
import math
import json
import random
import logging
import threading
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler

import ta
from ta.trend import macd, macd_signal, ADXIndicator
from ta.momentum import rsi, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, MarketData
from strategies.trading_types import OrderType

logger = logging.getLogger(__name__)

class RLParameterTuner:
    """Reinforcement Learning Parameter Tuner for automatic optimization"""
    
    def __init__(self, config: dict, param_state_path: str):
        self.config = config
        self.param_state_path = param_state_path
        self.episode_pnl = 0.0
        self.trade_count = 0
        self.best_pnl = -float("inf")
        self.best_params = {}
        self.losing_streak = 0
        self.winning_streak = 0
        self.original_order_size = self.config.get("order_size", 0.25)
        self.cooldown_until = 0
        self.load_state()
    
    def load_state(self):
        """Load saved parameter state"""
        if os.path.exists(self.param_state_path):
            try:
                with open(self.param_state_path, "r") as f:
                    state = json.load(f)
                self.best_pnl = state.get("best_pnl", -float("inf"))
                self.best_params = state.get("best_params", {})
                logger.info(f"[Tuner] Loaded {self.param_state_path}, best_pnl={self.best_pnl}")
            except Exception as e:
                logger.warning(f"[Tuner] Error loading state: {e}")
        else:
            logger.info(f"[Tuner] No existing {self.param_state_path}; fresh start.")
    
    def save_state(self):
        """Save current parameter state"""
        try:
            state = {"best_pnl": self.best_pnl, "best_params": self.best_params}
            with open(self.param_state_path, "w") as f:
                json.dump(state, f, indent=2)
            logger.info(f"[Tuner] Saved => {self.param_state_path}")
        except Exception as e:
            logger.warning(f"[Tuner] Error saving state: {e}")
    
    def on_trade_closed(self, trade_pnl: float):
        """Process closed trade for parameter optimization"""
        self.episode_pnl += trade_pnl
        self.trade_count += 1
        
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
            old_size = self.config.get("order_size", 0.25)
            self.config["order_size"] = old_size * 0.75
            self.cooldown_until = time.time() + 30
            logger.info(f"[Tuner] Losing streak => order_size {old_size} -> {self.config['order_size']:.4f}")
            self.losing_streak = 0
        
        if self.winning_streak >= 3:
            old_size = self.config.get("order_size", 0.25)
            new_size = min(old_size * 1.15, 2.0 * self.original_order_size)
            self.config["order_size"] = new_size
            logger.info(f"[Tuner] Winning streak => order_size {old_size} -> {new_size:.4f}")
            self.winning_streak = 0
    
    def evaluate_params(self):
        """Evaluate current parameters and optimize if needed"""
        if self.episode_pnl > self.best_pnl:
            self.best_pnl = self.episode_pnl
            self.best_params = {
                "stop_loss_pct": self.config.get("stop_loss_pct", 0.005),
                "take_profit_pct": self.config.get("take_profit_pct", 0.01),
                "trail_offset": self.config.get("trail_offset", 0.0025),
                "order_size": self.config.get("order_size", 0.25),
                "fast_ma": self.config.get("fast_ma", 5),
                "slow_ma": self.config.get("slow_ma", 15)
            }
            self.save_state()
        else:
            self.random_nudge()
        
        self.episode_pnl = 0.0
    
    def random_nudge(self):
        """Apply random parameter adjustments for exploration"""
        picks = ["stop_loss_pct", "take_profit_pct", "trail_offset", "order_size", "fast_ma", "slow_ma"]
        chosen = random.choice(picks)
        old_val = self.config.get(chosen, 0)
        
        if chosen == "stop_loss_pct":
            new_val = max(0.001, min(0.1, old_val + random.uniform(-0.005, 0.005)))
        elif chosen == "take_profit_pct":
            new_val = max(0.005, min(0.5, old_val + random.uniform(-0.01, 0.01)))
        elif chosen == "trail_offset":
            new_val = max(0.0, min(0.05, old_val + random.uniform(-0.005, 0.005)))
        elif chosen == "order_size":
            new_val = max(0.01, min(2.0, old_val + random.uniform(-0.05, 0.05)))
        elif chosen == "fast_ma":
            new_val = max(1, min(50, int(old_val + random.randint(-2, 2))))
        else:  # slow_ma
            new_val = max(5, min(200, int(old_val + random.randint(-2, 2))))
        
        self.config[chosen] = new_val
        logger.info(f"[Tuner] RandomNudge => {chosen}: {old_val} -> {new_val}")
    
    def is_in_cooldown(self) -> bool:
        """Check if tuner is in cooldown period"""
        return time.time() < self.cooldown_until


class TransformerPriceModel(nn.Module):
    """Transformer-based price prediction model with 12 features per bar"""
    
    def __init__(self, input_size_per_bar=12, lookback_bars=30, hidden_size=64, dropout_p=0.1):
        super().__init__()
        self.lookback_bars = lookback_bars
        self.input_size_per_bar = input_size_per_bar
        self.hidden_size = hidden_size
        
        self.embedding = nn.Linear(input_size_per_bar, hidden_size)
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
        self.reg_head = nn.Linear(hidden_size, 1)  # Price regression
        self.cls_head = nn.Linear(hidden_size, 3)  # Classification (SELL, HOLD, BUY)
    
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.lookback_bars, self.input_size_per_bar)
        x = self.embedding(x)
        
        # Transformer processing
        out = self.transformer(x, x)
        last_out = out[:, -1, :]  # Take last timestep
        last_out = self.dropout(last_out)
        
        # Dual heads for regression and classification
        reg_output = self.reg_head(last_out)
        cls_output = self.cls_head(last_out)
        
        return reg_output, cls_output


class EnhancedNeuralStrategy(BaseStrategy):
    """Enhanced Neural Network Strategy with RL Parameter Tuning"""
    
    def __init__(self, api, risk_manager, config=None):
        super().__init__(api, risk_manager, max_positions=3)
        
        # Configuration
        self.config = config or self._default_config()
        
        # Neural network parameters
        self.lookback_bars = self.config.get("nn_lookback_bars", 30)
        self.features_per_bar = 12  # Must match model input
        self.nn_hidden_size = self.config.get("nn_hidden_size", 64)
        self.nn_lr = self.config.get("nn_lr", 0.0003)
        self.synergy_conf_threshold = self.config.get("synergy_conf_threshold", 0.8)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[EnhancedNeural] Using device: {self.device}")
        
        # Initialize model
        self.model = TransformerPriceModel(
            input_size_per_bar=self.features_per_bar,
            lookback_bars=self.lookback_bars,
            hidden_size=self.nn_hidden_size,
            dropout_p=0.1
        ).to(self.device)
        
        self.optimizer = Adam(self.model.parameters(), lr=self.nn_lr)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.95)
        
        # Data storage
        self.hist_data = pd.DataFrame(columns=[
            "time", "price", "volume", "vol_ma", "fast_ma", "slow_ma", "rsi",
            "macd_hist", "bb_high", "bb_low", "stoch_k", "stoch_d", "adx", "atr"
        ])
        self.training_data = []
        self.trade_pnls = []
        
        # Model persistence
        self.model_checkpoint_path = "enhanced_neural_model.pth"
        self._try_load_model()
        
        # RL Parameter Tuner
        self.param_state_path = "enhanced_neural_params.json"
        self.tuner = RLParameterTuner(self.config, self.param_state_path)
        
        # Training executor
        self.training_executor = ThreadPoolExecutor(max_workers=1)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # State tracking
        self.warmup_done = False
        self.warmup_start = time.time()
        self.warmup_duration = 20.0
        self.last_trade_time = 0
        self.hold_counter = 0
        
        logger.info("Enhanced Neural Network Strategy initialized")
    
    def _default_config(self) -> dict:
        """Default configuration for the strategy"""
        return {
            "fast_ma": 5,
            "slow_ma": 15,
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "boll_period": 20,
            "boll_stddev": 2.0,
            "stop_loss_pct": 0.005,
            "take_profit_pct": 0.01,
            "trail_offset": 0.0025,
            "order_size": 0.25,
            "min_trade_interval": 60,
            "nn_lookback_bars": 30,
            "nn_hidden_size": 64,
            "nn_lr": 0.0003,
            "synergy_conf_threshold": 0.8
        }
    
    def _try_load_model(self):
        """Load saved model if available"""
        if os.path.exists(self.model_checkpoint_path):
            try:
                state_dict = torch.load(self.model_checkpoint_path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"[Model] Loaded => {self.model_checkpoint_path}")
            except Exception as e:
                logger.warning(f"[Model] Load error: {e}")
        else:
            logger.info("[Model] No checkpoint found; starting fresh.")
    
    def _save_model(self):
        """Save current model state"""
        try:
            torch.save(self.model.state_dict(), self.model_checkpoint_path)
            logger.info(f"[Model] Saved => {self.model_checkpoint_path}")
        except Exception as e:
            logger.warning(f"[Model] Save error: {e}")
    
    def compute_indicators(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Compute technical indicators for the dataframe"""
        required_length = max(
            self.config.get("slow_ma", 15),
            self.config.get("boll_period", 20),
            self.config.get("macd_slow", 26)
        )
        
        if len(df) < required_length:
            return None
        
        # Clean data
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Volume moving average
        df["vol_ma"] = df["volume"].rolling(20).mean().bfill().fillna(1.0)
        
        # Price data
        close = df["price"]
        
        # Moving averages
        df["fast_ma"] = close.rolling(self.config.get("fast_ma", 5)).mean()
        df["slow_ma"] = close.rolling(self.config.get("slow_ma", 15)).mean()
        
        # RSI
        df["rsi"] = rsi(close, window=self.config.get("rsi_period", 14))
        
        # MACD
        macd_val = macd(close, self.config.get("macd_slow", 26), self.config.get("macd_fast", 12))
        macd_sig = macd_signal(close, self.config.get("macd_slow", 26), 
                              self.config.get("macd_fast", 12), self.config.get("macd_signal", 9))
        df["macd_hist"] = macd_val - macd_sig
        
        # Bollinger Bands
        bb = BollingerBands(close, self.config.get("boll_period", 20), self.config.get("boll_stddev", 2.0))
        df["bb_high"] = bb.bollinger_hband()
        df["bb_low"] = bb.bollinger_lband()
        
        # High/Low for other indicators
        df["high"] = df["price"].rolling(2).max()
        df["low"] = df["price"].rolling(2).min()
        
        # Stochastic Oscillator
        try:
            stoch = StochasticOscillator(high=df["high"], low=df["low"], close=close, window=14, smooth_window=3)
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()
        except Exception as e:
            logger.warning(f"[Stochastic] Error: {e}. Setting default values.")
            df["stoch_k"] = 50.0
            df["stoch_d"] = 50.0
        
        # ADX
        try:
            adx_ind = ADXIndicator(high=df["high"], low=df["low"], close=close, window=14)
            df["adx"] = adx_ind.adx()
        except Exception as e:
            logger.warning(f"[ADX] Error: {e}. Setting adx=0.")
            df["adx"] = 0.0
        
        # ATR
        try:
            atr_ind = AverageTrueRange(high=df["high"], low=df["low"], close=close, window=14)
            df["atr"] = atr_ind.average_true_range()
        except Exception as e:
            logger.warning(f"[ATR] Error: {e}. Setting atr=0.005.")
            df["atr"] = 0.005
        
        # Fill missing values
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        df.fillna(0.0, inplace=True)
        
        return df.iloc[-1]
    
    def build_input_features(self, block: pd.DataFrame) -> List[float]:
        """Build input features for neural network (12 features per bar)"""
        features = []
        
        for _, row in block.iterrows():
            # Bollinger Band position
            bb_high = row.get("bb_high", 0.0)
            bb_low = row.get("bb_low", 0.0)
            bb_range = (bb_high - bb_low) if (bb_high - bb_low) != 0 else 1e-9
            bb_position = (row["price"] - bb_low) / bb_range
            
            # Volume factor
            vol_ma = row.get("vol_ma", 1.0)
            vol_factor = (row["volume"] / max(vol_ma, 1e-9)) - 1
            
            # ATR and BB range
            atr = row.get("atr", 0.005)
            bb_range_norm = bb_high - bb_low
            
            # 12 features per bar
            bar_features = [
                row["price"],
                row.get("fast_ma", row["price"]),
                row.get("slow_ma", row["price"]),
                row.get("rsi", 50),
                row.get("macd_hist", 0),
                bb_position,
                vol_factor,
                row.get("stoch_k", 0.0),
                row.get("stoch_d", 0.0),
                row.get("adx", 0.0),
                atr,
                bb_range_norm
            ]
            
            # Check for invalid values
            if any(math.isnan(x) or math.isinf(x) for x in bar_features):
                return []
            
            features.extend(bar_features)
        
        return features
    
    def store_training_data(self):
        """Store training data for model learning"""
        if len(self.hist_data) < (self.lookback_bars + 2):
            return
        
        # Get historical block and future price
        block = self.hist_data.iloc[-(self.lookback_bars + 2):-2]
        if len(block) < self.lookback_bars:
            return
        
        last_bar = self.hist_data.iloc[-2]
        future_bar = self.hist_data.iloc[-1]
        
        # Build features
        features = self.build_input_features(block)
        if not features:
            return
        
        # Calculate price change and classification
        price_diff = (future_bar["price"] - last_bar["price"]) / max(last_bar["price"], 1e-9)
        cls_label = 2 if price_diff > 0.005 else (0 if price_diff < -0.005 else 1)
        
        # Store training sample
        self.training_data.append((features, future_bar["price"], cls_label))
        
        # Limit training data size
        if len(self.training_data) > 2000:
            self.training_data.pop(0)
    
    def train_model_batch(self, batch_size=16):
        """Train model with mini-batch"""
        if len(self.training_data) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.training_data, batch_size)
        X_features, Y_reg, Y_cls = [], [], []
        
        for features, next_price, cls_label in batch:
            X_features.append(features)
            Y_reg.append(next_price)
            Y_cls.append(cls_label)
        
        # Convert to numpy and check for invalid values
        X_features = np.array(X_features)
        if np.isnan(X_features).any() or np.isinf(X_features).any():
            return
        
        # Scale features
        self.scaler.fit(X_features)
        X_scaled = self.scaler.transform(X_features)
        
        # Convert to tensors
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        y_reg = torch.tensor(Y_reg, dtype=torch.float32, device=self.device).view(-1, 1)
        y_cls = torch.tensor(Y_cls, dtype=torch.long, device=self.device)
        
        # Training step
        self.model.train()
        reg_out, cls_out = self.model(x_tensor)
        
        # Calculate losses
        loss_reg = nn.MSELoss()(reg_out, y_reg)
        loss_cls = nn.CrossEntropyLoss()(cls_out, y_cls)
        total_loss = loss_reg + loss_cls
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        logger.info(f"[Training] total={total_loss.item():.4f}, reg={loss_reg.item():.4f}, cls={loss_cls.item():.4f}")
    
    def async_training(self):
        """Asynchronous training to prevent blocking"""
        self.training_executor.submit(self.train_model_batch, 16)
    
    def generate_signal(self, market_data: MarketData) -> TradingSignal:
        """Generate trading signal using neural network"""
        try:
            # Check warmup period
            if not self.warmup_done:
                remaining = self.warmup_duration - (time.time() - self.warmup_start)
                if remaining > 0:
                    return TradingSignal(SignalType.HOLD, 0.5, "Warmup period")
                else:
                    self.warmup_done = True
            
            # Check tuner cooldown
            if self.tuner.is_in_cooldown():
                return TradingSignal(SignalType.HOLD, 0.5, "Tuner cooldown")
            
            # Add new data point
            now_str = datetime.utcnow().isoformat()
            new_row = pd.DataFrame([[
                now_str, market_data.price, market_data.volume,
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            ]], columns=self.hist_data.columns)
            
            self.hist_data = pd.concat([self.hist_data, new_row], ignore_index=True)
            
            # Limit data size
            if len(self.hist_data) > 2000:
                self.hist_data = self.hist_data.iloc[-2000:]
            
            # Compute indicators
            row = self.compute_indicators(self.hist_data)
            if row is None:
                return TradingSignal(SignalType.HOLD, 0.5, "Insufficient data for indicators")
            
            # Store training data and train asynchronously
            self.store_training_data()
            self.async_training()
            
            # Check minimum trade interval
            if time.time() - self.last_trade_time < self.config.get("min_trade_interval", 60):
                return TradingSignal(SignalType.HOLD, 0.5, "Trade interval not met")
            
            # Generate neural network prediction
            if len(self.hist_data) >= self.lookback_bars:
                features = self.build_input_features(self.hist_data.iloc[-self.lookback_bars:])
                
                if len(features) == (self.lookback_bars * self.features_per_bar):
                    # Prepare input
                    X_input = np.array([features])
                    self.scaler.fit(X_input)
                    X_scaled = self.scaler.transform(X_input)
                    x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
                    
                    # Model inference
                    self.model.eval()
                    with torch.no_grad():
                        reg_out, cls_out = self.model(x_tensor)
                    
                    pred_price = reg_out[0, 0].item()
                    signal = self._final_inference(row, pred_price, cls_out[0])
                    
                    if signal != "HOLD":
                        self.last_trade_time = time.time()
                    
                    # Convert to TradingSignal
                    if signal == "BUY":
                        return TradingSignal(SignalType.BUY, 0.8, "Neural network BUY signal")
                    elif signal == "SELL":
                        return TradingSignal(SignalType.SELL, 0.8, "Neural network SELL signal")
            
            return TradingSignal(SignalType.HOLD, 0.5, "Neural network HOLD signal")
            
        except Exception as e:
            logger.error(f"Error generating neural signal: {e}")
            return TradingSignal(SignalType.HOLD, 0.5, f"Error: {e}")
    
    def _final_inference(self, row: pd.Series, pred_reg: float, pred_cls: torch.Tensor) -> str:
        """Final inference combining regression and classification"""
        # Classification confidence
        soft_probs = F.softmax(pred_cls, dim=0)
        cls_conf, cls_idx = torch.max(soft_probs, dim=0)
        cls_conf = cls_conf.item()
        cls_idx = cls_idx.item()
        
        nn_cls_signal = "SELL" if cls_idx == 0 else ("BUY" if cls_idx == 2 else "HOLD")
        
        # Regression signal
        current_price = row["price"]
        price_diff = (pred_reg - current_price) / max(current_price, 1e-9)
        nn_reg_signal = "BUY" if price_diff > 0 else ("SELL" if price_diff < 0 else "HOLD")
        
        # Adaptive threshold based on ATR
        atr = row.get("atr", 0.005)
        adaptive_threshold = max(0.005, 0.5 * atr)
        
        # Market regime detection
        adx = row.get("adx", 0)
        regime = "trending" if adx > 25 else "ranging"
        adaptive_threshold *= 0.8 if regime == "trending" else 1.2
        
        logger.info(f"[Inference] price_diff={price_diff:.4f}, threshold={adaptive_threshold:.4f}, cls_conf={cls_conf:.4f}")
        
        # Decision logic
        if abs(price_diff) < adaptive_threshold or cls_conf < self.synergy_conf_threshold:
            decision = "HOLD"
        else:
            signals = [nn_cls_signal, nn_reg_signal]
            decision = "BUY" if signals.count("BUY") > signals.count("SELL") else "SELL"
        
        # Override persistent HOLD
        if decision == "HOLD":
            self.hold_counter += 1
            if self.hold_counter >= 10:
                signals = [nn_cls_signal, nn_reg_signal]
                decision = "BUY" if signals.count("BUY") >= signals.count("SELL") else "SELL"
                logger.info(f"[Override] HOLD persisted 10 cycles; forcing decision: {decision}")
        else:
            self.hold_counter = 0
        
        return decision
    
    def on_trade_closed(self, trade_pnl: float):
        """Handle trade closure for RL tuning"""
        self.trade_pnls.append(trade_pnl)
        self.tuner.on_trade_closed(trade_pnl)
    
    def save_state(self):
        """Save strategy state"""
        self._save_model()
        self.tuner.save_state()
    
    def get_strategy_info(self) -> dict:
        """Get strategy information"""
        return {
            "name": "Enhanced Neural Network Strategy",
            "type": "AI/ML",
            "features": [
                "Transformer neural network",
                "RL parameter tuning",
                "12 technical indicators",
                "Adaptive thresholds",
                "Synergy confidence"
            ],
            "parameters": {
                "lookback_bars": self.lookback_bars,
                "features_per_bar": self.features_per_bar,
                "synergy_threshold": self.synergy_conf_threshold,
                "model_device": str(self.device)
            }
        }

