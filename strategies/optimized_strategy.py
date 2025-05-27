"""
Optimized Strategy for Hyperliquid Trading.
Implements advanced market regime detection and adaptive parameter selection.
"""

import os
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

class OptimizedStrategy:
    """
    Optimized trading strategy with market regime detection and adaptive parameters.
    """
    
    def __init__(self, adapter, position_manager, risk_manager, trading_mode_manager):
        """
        Initialize the optimized strategy.
        
        Args:
            adapter: HyperliquidAdapter instance
            position_manager: PositionManager instance
            risk_manager: RiskManager instance
            trading_mode_manager: TradingModeManager instance
        """
        self.adapter = adapter
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        self.trading_mode_manager = trading_mode_manager
        self.logger = logging.getLogger("OptimizedStrategy")
        
        # Market regimes
        self.REGIME_BULLISH = "BULLISH"
        self.REGIME_BEARISH = "BEARISH"
        self.REGIME_RANGING = "RANGING"
        self.REGIME_VOLATILE = "VOLATILE"
        
        # Signal types
        self.SIGNAL_LONG = "LONG"
        self.SIGNAL_SHORT = "SHORT"
        self.SIGNAL_NEUTRAL = "NEUTRAL"
        self.SIGNAL_EXIT = "EXIT"
        
        # Strategy parameters
        self.params = {
            "default": {
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "ema_short_period": 9,
                "ema_medium_period": 21,
                "ema_long_period": 50,
                "atr_period": 14,
                "atr_multiplier": 2.0,
                "volume_ma_period": 20,
                "funding_rate_threshold": 0.0001,
                "min_rrr": 2.0
            },
            self.REGIME_BULLISH: {
                "rsi_overbought": 80,
                "rsi_oversold": 40,
                "atr_multiplier": 2.5,
                "min_rrr": 1.5
            },
            self.REGIME_BEARISH: {
                "rsi_overbought": 60,
                "rsi_oversold": 20,
                "atr_multiplier": 2.5,
                "min_rrr": 1.5
            },
            self.REGIME_RANGING: {
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "atr_multiplier": 1.5,
                "min_rrr": 2.0
            },
            self.REGIME_VOLATILE: {
                "rsi_overbought": 75,
                "rsi_oversold": 25,
                "atr_multiplier": 3.0,
                "min_rrr": 2.5
            }
        }
    
    def get_parameters(self, regime: str) -> Dict[str, Any]:
        """
        Get parameters for a specific market regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Dict containing parameters
        """
        # Start with default parameters
        result = self.params["default"].copy()
        
        # Override with regime-specific parameters
        if regime in self.params:
            for key, value in self.params[regime].items():
                result[key] = value
        
        # Apply mode-specific adjustments
        mode = self.trading_mode_manager.get_current_mode()
        mode_settings = self.trading_mode_manager.get_mode_settings()
        
        if mode_settings:
            # Adjust risk parameters based on mode
            if "risk_multiplier" in mode_settings:
                result["atr_multiplier"] *= mode_settings["risk_multiplier"]
            
            if "min_rrr" in mode_settings:
                result["min_rrr"] = mode_settings["min_rrr"]
        
        return result
    
    def detect_market_regime(self, symbol: str) -> str:
        """
        Detect the current market regime.
        
        Args:
            symbol: The symbol to detect market regime for
            
        Returns:
            Market regime (BULLISH, BEARISH, RANGING, VOLATILE)
        """
        try:
            # Get historical data
            klines = self.adapter.get_klines(symbol, interval="1h", limit=100)
            
            if "error" in klines:
                self.logger.error(f"Error getting klines: {klines['error']}")
                return self.REGIME_RANGING  # Default to ranging if error
            
            # Extract close prices
            closes = []
            for candle in klines.get("data", []):
                if isinstance(candle, list) and len(candle) >= 5:
                    closes.append(float(candle[4]))  # Close price
            
            if not closes:
                self.logger.error("No close prices found")
                return self.REGIME_RANGING  # Default to ranging if no data
            
            # Convert to numpy array for calculations
            closes = np.array(closes)
            
            # Calculate indicators
            ema20 = self._calculate_ema(closes, 20)
            ema50 = self._calculate_ema(closes, 50)
            ema200 = self._calculate_ema(closes, 200)
            
            # Calculate volatility
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * np.sqrt(24)  # Annualized volatility
            
            # Calculate trend strength
            adx = self._calculate_adx(klines.get("data", []))
            
            # Determine regime
            if volatility > 0.08:  # High volatility threshold
                return self.REGIME_VOLATILE
            
            if adx > 25:  # Strong trend
                if ema20[-1] > ema50[-1] and ema50[-1] > ema200[-1]:
                    return self.REGIME_BULLISH
                elif ema20[-1] < ema50[-1] and ema50[-1] < ema200[-1]:
                    return self.REGIME_BEARISH
            
            # Default to ranging
            return self.REGIME_RANGING
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return self.REGIME_RANGING  # Default to ranging if error
    
    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Generate a trading signal.
        
        Args:
            symbol: The symbol to generate signal for
            
        Returns:
            Dict containing signal information
        """
        try:
            # Detect market regime
            regime = self.detect_market_regime(symbol)
            self.logger.info(f"Detected market regime: {regime}")
            
            # Get parameters for the regime
            params = self.get_parameters(regime)
            
            # Get historical data
            klines = self.adapter.get_klines(symbol, interval="1h", limit=100)
            
            if "error" in klines:
                self.logger.error(f"Error getting klines: {klines['error']}")
                return {"signal": self.SIGNAL_NEUTRAL, "regime": regime}
            
            # Extract OHLCV data
            candles = klines.get("data", [])
            
            if not candles:
                self.logger.error("No candles found")
                return {"signal": self.SIGNAL_NEUTRAL, "regime": regime}
            
            # Prepare data arrays
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            
            for candle in candles:
                if isinstance(candle, list) and len(candle) >= 6:
                    opens.append(float(candle[1]))
                    highs.append(float(candle[2]))
                    lows.append(float(candle[3]))
                    closes.append(float(candle[4]))
                    volumes.append(float(candle[5]))
            
            if not closes:
                self.logger.error("No valid candle data found")
                return {"signal": self.SIGNAL_NEUTRAL, "regime": regime}
            
            # Convert to numpy arrays
            opens = np.array(opens)
            highs = np.array(highs)
            lows = np.array(lows)
            closes = np.array(closes)
            volumes = np.array(volumes)
            
            # Calculate indicators
            rsi = self._calculate_rsi(closes, params["rsi_period"])
            ema_short = self._calculate_ema(closes, params["ema_short_period"])
            ema_medium = self._calculate_ema(closes, params["ema_medium_period"])
            ema_long = self._calculate_ema(closes, params["ema_long_period"])
            atr = self._calculate_atr(highs, lows, closes, params["atr_period"])
            
            # Get funding rate
            funding_rate_result = self.adapter.get_funding_rate(symbol)
            funding_rate = funding_rate_result.get("funding_rate", 0) if "success" in funding_rate_result else 0
            
            # Get current price
            ticker = self.adapter.get_ticker(symbol)
            current_price = ticker.get("data", {}).get("last_price", closes[-1]) if "success" in ticker else closes[-1]
            
            # Generate signal
            signal = self.SIGNAL_NEUTRAL
            entry_price = current_price
            stop_loss_price = 0
            take_profit_price = 0
            
            # Check for long signal
            long_condition = (
                rsi[-1] < params["rsi_oversold"] and
                ema_short[-1] > ema_short[-2] and  # Short EMA turning up
                volumes[-1] > np.mean(volumes[-params["volume_ma_period"]:]) and  # Above average volume
                (funding_rate < -params["funding_rate_threshold"] if regime != self.REGIME_BULLISH else True)  # Negative funding in non-bullish regimes
            )
            
            # Check for short signal
            short_condition = (
                rsi[-1] > params["rsi_overbought"] and
                ema_short[-1] < ema_short[-2] and  # Short EMA turning down
                volumes[-1] > np.mean(volumes[-params["volume_ma_period"]:]) and  # Above average volume
                (funding_rate > params["funding_rate_threshold"] if regime != self.REGIME_BEARISH else True)  # Positive funding in non-bearish regimes
            )
            
            # Generate long signal
            if long_condition:
                signal = self.SIGNAL_LONG
                entry_price = current_price
                stop_loss_price = entry_price - (atr[-1] * params["atr_multiplier"])
                take_profit_price = entry_price + (atr[-1] * params["atr_multiplier"] * params["min_rrr"])
            
            # Generate short signal
            elif short_condition:
                signal = self.SIGNAL_SHORT
                entry_price = current_price
                stop_loss_price = entry_price + (atr[-1] * params["atr_multiplier"])
                take_profit_price = entry_price - (atr[-1] * params["atr_multiplier"] * params["min_rrr"])
            
            # Check for exit signal for existing positions
            positions_result = self.adapter.get_positions()
            
            if "success" in positions_result:
                positions = positions_result.get("data", [])
                
                for position in positions:
                    if position.get("coin") == symbol:
                        size = float(position.get("szi", 0))
                        
                        if size != 0:
                            is_long = size > 0
                            
                            # Exit long position
                            if is_long and (
                                rsi[-1] > params["rsi_overbought"] or
                                ema_short[-1] < ema_medium[-1]
                            ):
                                signal = self.SIGNAL_EXIT
                            
                            # Exit short position
                            elif not is_long and (
                                rsi[-1] < params["rsi_oversold"] or
                                ema_short[-1] > ema_medium[-1]
                            ):
                                signal = self.SIGNAL_EXIT
            
            return {
                "signal": signal,
                "regime": regime,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "indicators": {
                    "rsi": rsi[-1],
                    "ema_short": ema_short[-1],
                    "ema_medium": ema_medium[-1],
                    "ema_long": ema_long[-1],
                    "atr": atr[-1],
                    "funding_rate": funding_rate
                },
                "parameters": params
            }
        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            return {"signal": self.SIGNAL_NEUTRAL, "regime": self.REGIME_RANGING}
    
    def execute(self, symbol: str) -> Dict[str, Any]:
        """
        Execute the strategy.
        
        Args:
            symbol: The symbol to execute strategy for
            
        Returns:
            Dict containing execution result
        """
        try:
            # Check if trading is allowed in current mode
            mode = self.trading_mode_manager.get_current_mode()
            mode_settings = self.trading_mode_manager.get_mode_settings()
            
            if mode_settings and not mode_settings.get("allow_trading", True):
                self.logger.info(f"Trading not allowed in {mode.name} mode")
                return {"success": False, "message": f"Trading not allowed in {mode.name} mode"}
            
            # Generate signal
            signal_result = self.generate_signal(symbol)
            
            if "signal" not in signal_result:
                self.logger.error("No signal generated")
                return {"success": False, "message": "No signal generated"}
            
            signal = signal_result["signal"]
            regime = signal_result["regime"]
            
            self.logger.info(f"Generated signal: {signal} in {regime} regime")
            
            # Execute signal
            if signal == self.SIGNAL_LONG:
                # Get account info for position sizing
                account_info = self.adapter.get_account_info()
                
                if "error" in account_info:
                    self.logger.error(f"Error getting account info: {account_info['error']}")
                    return {"success": False, "message": f"Error getting account info: {account_info['error']}"}
                
                equity = account_info.get("equity", 0)
                
                # Calculate position size
                entry_price = signal_result["entry_price"]
                stop_loss_price = signal_result["stop_loss_price"]
                take_profit_price = signal_result["take_profit_price"]
                
                # Validate risk-reward ratio
                rrr = self.risk_manager.calculate_risk_reward_ratio(
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price
                )
                
                if not self.risk_manager.validate_risk_reward_ratio(
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price
                ):
                    self.logger.info(f"Risk-reward ratio {rrr} below minimum threshold")
                    return {"success": False, "message": f"Risk-reward ratio {rrr} below minimum threshold"}
                
                # Calculate position size
                size = self.risk_manager.calculate_position_size(
                    account_equity=equity,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    symbol=symbol
                )
                
                # Open long position
                result = self.position_manager.open_position(
                    symbol=symbol,
                    is_long=True,
                    size=size,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price
                )
                
                return {
                    "success": "error" not in result,
                    "message": "Long position opened successfully" if "error" not in result else result.get("error", "Unknown error"),
                    "position": "long",
                    "size": size,
                    "entry_price": entry_price,
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": take_profit_price,
                    "regime": regime
                }
            
            elif signal == self.SIGNAL_SHORT:
                # Get account info for position sizing
                account_info = self.adapter.get_account_info()
                
                if "error" in account_info:
                    self.logger.error(f"Error getting account info: {account_info['error']}")
                    return {"success": False, "message": f"Error getting account info: {account_info['error']}"}
                
                equity = account_info.get("equity", 0)
                
                # Calculate position size
                entry_price = signal_result["entry_price"]
                stop_loss_price = signal_result["stop_loss_price"]
                take_profit_price = signal_result["take_profit_price"]
                
                # Validate risk-reward ratio
                rrr = self.risk_manager.calculate_risk_reward_ratio(
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price
                )
                
                if not self.risk_manager.validate_risk_reward_ratio(
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price
                ):
                    self.logger.info(f"Risk-reward ratio {rrr} below minimum threshold")
                    return {"success": False, "message": f"Risk-reward ratio {rrr} below minimum threshold"}
                
                # Calculate position size
                size = self.risk_manager.calculate_position_size(
                    account_equity=equity,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    symbol=symbol
                )
                
                # Open short position
                result = self.position_manager.open_position(
                    symbol=symbol,
                    is_long=False,
                    size=size,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price
                )
                
                return {
                    "success": "error" not in result,
                    "message": "Short position opened successfully" if "error" not in result else result.get("error", "Unknown error"),
                    "position": "short",
                    "size": size,
                    "entry_price": entry_price,
                    "stop_loss_price": stop_loss_price,
                    "take_profit_price": take_profit_price,
                    "regime": regime
                }
            
            elif signal == self.SIGNAL_EXIT:
                # Close position
                result = self.position_manager.close_position(symbol=symbol)
                
                return {
                    "success": "error" not in result,
                    "message": "Position closed successfully" if "error" not in result else result.get("error", "Unknown error"),
                    "position": "exit",
                    "regime": regime
                }
            
            else:
                return {
                    "success": True,
                    "message": "No action taken",
                    "position": "neutral",
                    "regime": regime
                }
        except Exception as e:
            self.logger.error(f"Error executing strategy: {e}")
            return {"success": False, "message": f"Error executing strategy: {e}"}
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Array of prices
            period: RSI period
            
        Returns:
            Array of RSI values
        """
        # Calculate price changes
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)
        
        # Calculate RSI
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            rs = up/down if down != 0 else 0
            rsi[i] = 100. - 100./(1. + rs)
        
        return rsi
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Array of prices
            period: EMA period
            
        Returns:
            Array of EMA values
        """
        ema = np.zeros_like(prices)
        ema[:period] = np.mean(prices[:period])
        
        # Calculate multiplier
        multiplier = 2.0 / (period + 1)
        
        # Calculate EMA
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Average True Range.
        
        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of close prices
            period: ATR period
            
        Returns:
            Array of ATR values
        """
        # Calculate true range
        tr = np.zeros(len(closes))
        tr[0] = highs[0] - lows[0]
        
        for i in range(1, len(closes)):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
        
        # Calculate ATR
        atr = np.zeros_like(closes)
        atr[:period] = np.mean(tr[:period])
        
        for i in range(period, len(closes)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        
        return atr
    
    def _calculate_adx(self, candles: List[List[float]], period: int = 14) -> float:
        """
        Calculate Average Directional Index.
        
        Args:
            candles: List of candles [timestamp, open, high, low, close, volume]
            period: ADX period
            
        Returns:
            ADX value
        """
        if len(candles) < period + 1:
            return 0
        
        # Extract high, low, close
        highs = []
        lows = []
        closes = []
        
        for candle in candles:
            if isinstance(candle, list) and len(candle) >= 5:
                highs.append(float(candle[2]))
                lows.append(float(candle[3]))
                closes.append(float(candle[4]))
        
        if len(highs) < period + 1:
            return 0
        
        # Convert to numpy arrays
        highs = np.array(highs)
        lows = np.array(lows)
        closes = np.array(closes)
        
        # Calculate +DM and -DM
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]
        
        plus_dm = np.zeros(len(up_move))
        minus_dm = np.zeros(len(down_move))
        
        for i in range(len(up_move)):
            if up_move[i] > down_move[i] and up_move[i] > 0:
                plus_dm[i] = up_move[i]
            else:
                plus_dm[i] = 0
            
            if down_move[i] > up_move[i] and down_move[i] > 0:
                minus_dm[i] = down_move[i]
            else:
                minus_dm[i] = 0
        
        # Calculate true range
        tr = np.zeros(len(closes) - 1)
        for i in range(len(tr)):
            tr[i] = max(
                highs[i+1] - lows[i+1],
                abs(highs[i+1] - closes[i]),
                abs(lows[i+1] - closes[i])
            )
        
        # Calculate smoothed values
        tr_period = tr[:period].sum()
        plus_dm_period = plus_dm[:period].sum()
        minus_dm_period = minus_dm[:period].sum()
        
        tr_smooth = [tr_period]
        plus_dm_smooth = [plus_dm_period]
        minus_dm_smooth = [minus_dm_period]
        
        for i in range(period, len(tr)):
            tr_smooth.append(tr_smooth[-1] - tr_smooth[-1]/period + tr[i])
            plus_dm_smooth.append(plus_dm_smooth[-1] - plus_dm_smooth[-1]/period + plus_dm[i])
            minus_dm_smooth.append(minus_dm_smooth[-1] - minus_dm_smooth[-1]/period + minus_dm[i])
        
        # Calculate +DI and -DI
        plus_di = 100 * (np.array(plus_dm_smooth) / np.array(tr_smooth))
        minus_di = 100 * (np.array(minus_dm_smooth) / np.array(tr_smooth))
        
        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = np.mean(dx[-period:])
        
        return adx
