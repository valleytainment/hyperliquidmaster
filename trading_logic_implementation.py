"""
Enhanced implementation of the generate_signals and execute_trades methods
for the HyperliquidMaster trading bot.
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

from core.trading_mode import TradingMode
from strategies.master_omni_overlord import MasterOmniOverlordStrategy
from strategies.strategy_integration import StrategyIntegration

async def generate_signals(self):
    """Generate trading signals using advanced strategy logic."""
    self.logger.info("Generating trading signals...")
    
    # Get symbols to monitor
    symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
    
    # Initialize signals dictionary if not exists
    if not hasattr(self, 'signals') or self.signals is None:
        self.signals = {}
    
    # Initialize strategy if not exists
    if not hasattr(self, 'strategy') or self.strategy is None:
        self.strategy = MasterOmniOverlordStrategy(self.config, self.logger)
    
    for symbol in symbols:
        try:
            if symbol not in self.market_data:
                self.logger.warning(f"No market data available for {symbol}, skipping signal generation")
                continue
            
            # Get market data for the symbol
            market_data = self.market_data[symbol]
            
            # Check if we have historical data
            if 'historical_data' not in market_data or market_data['historical_data'] is None:
                # Fetch historical data if not available
                try:
                    historical_data = await self.exchange.fetch_historical_data(symbol)
                    if historical_data is not None and not isinstance(historical_data, dict):
                        # Convert to DataFrame if it's not already
                        if not isinstance(historical_data, pd.DataFrame):
                            # Assume it's a list of dictionaries
                            historical_data = pd.DataFrame(historical_data)
                        
                        # Add to market data
                        self.market_data[symbol]['historical_data'] = historical_data
                    else:
                        self.logger.warning(f"Could not fetch historical data for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            
            # Prepare market data with indicators
            enhanced_market_data = StrategyIntegration.prepare_market_data(self.market_data[symbol])
            self.market_data[symbol] = enhanced_market_data
            
            # Detect market regime if we have historical data
            if 'historical_data' in enhanced_market_data and enhanced_market_data['historical_data'] is not None:
                try:
                    market_regime = self.strategy.detect_market_regime(enhanced_market_data['historical_data'])
                    self.logger.info(f"Market regime for {symbol}: {market_regime}")
                except Exception as e:
                    self.logger.error(f"Error detecting market regime for {symbol}: {e}")
                    market_regime = "unknown"
            else:
                market_regime = "unknown"
            
            # Adjust strategy parameters based on market conditions
            self.strategy.adjust_adaptive_parameters()
            self.strategy.adjust_strategy_weights()
            
            # Generate signal
            signal = {
                "symbol": symbol,
                "timestamp": time.time(),
                "price": enhanced_market_data.get("price", 0),
                "signal": "NEUTRAL",  # Default signal
                "strength": 0,
                "confidence": 0,
                "regime": market_regime,
                "indicators": enhanced_market_data.get("indicators", {})
            }
            
            # Get position size if we have a signal
            current_position = self.positions.get(symbol, {"size": 0})
            current_position_size = current_position.get("size", 0)
            
            # Generate signals from sub-strategies
            try:
                # Get signals from each sub-strategy
                tc_signal = self.strategy.triple_confluence.generate_signal(enhanced_market_data)
                ou_signal = self.strategy.oracle_update.generate_signal(enhanced_market_data)
                
                # Combine signals using strategy weights
                tc_weight = self.strategy.strategy_weights.get("triple_confluence", 0.5)
                ou_weight = self.strategy.strategy_weights.get("oracle_update", 0.5)
                
                # Calculate weighted signal strength
                signal_strength = (
                    tc_signal.get("strength", 0) * tc_weight +
                    ou_signal.get("strength", 0) * ou_weight
                )
                
                # Calculate weighted confidence
                confidence = (
                    tc_signal.get("confidence", 0) * tc_weight +
                    ou_signal.get("confidence", 0) * ou_weight
                )
                
                # Determine signal direction
                if signal_strength > self.strategy.adaptive_params.get("signal_threshold", 0.7):
                    signal["signal"] = "LONG"
                    signal["strength"] = signal_strength
                    signal["confidence"] = confidence
                elif signal_strength < -self.strategy.adaptive_params.get("signal_threshold", 0.7):
                    signal["signal"] = "SHORT"
                    signal["strength"] = abs(signal_strength)
                    signal["confidence"] = confidence
                else:
                    signal["signal"] = "NEUTRAL"
                    signal["strength"] = abs(signal_strength)
                    signal["confidence"] = confidence
                
                # Calculate target position size
                if signal["signal"] != "NEUTRAL":
                    # Get current price and calculate stop loss
                    current_price = enhanced_market_data.get("price", 0)
                    
                    # Calculate stop loss based on ATR or support/resistance
                    if "indicators" in enhanced_market_data and enhanced_market_data["indicators"] is not None:
                        indicators = enhanced_market_data["indicators"]
                        if signal["signal"] == "LONG":
                            # For long positions, use support level or percentage-based stop
                            if "support_level" in indicators and indicators["support_level"] is not None:
                                stop_loss = indicators["support_level"]
                            else:
                                stop_loss = current_price * 0.98  # 2% default stop loss
                        else:
                            # For short positions, use resistance level or percentage-based stop
                            if "resistance_level" in indicators and indicators["resistance_level"] is not None:
                                stop_loss = indicators["resistance_level"]
                            else:
                                stop_loss = current_price * 1.02  # 2% default stop loss
                    else:
                        # Default stop loss if no indicators
                        stop_loss = current_price * (0.98 if signal["signal"] == "LONG" else 1.02)
                    
                    # Calculate position size
                    target_position_size = self.strategy.calculate_position_size(
                        symbol=symbol,
                        entry_price=current_price,
                        stop_loss=stop_loss
                    )
                    
                    # Adjust for direction
                    if signal["signal"] == "SHORT":
                        target_position_size = -target_position_size
                    
                    signal["target_position_size"] = target_position_size
                    signal["stop_loss"] = stop_loss
                    
                    # Calculate take profit based on risk-reward ratio
                    risk = abs(current_price - stop_loss)
                    reward_ratio = 2.0  # Default 1:2 risk-reward ratio
                    
                    if signal["signal"] == "LONG":
                        take_profit = current_price + (risk * reward_ratio)
                    else:
                        take_profit = current_price - (risk * reward_ratio)
                    
                    signal["take_profit"] = take_profit
                
                self.logger.info(f"Generated signal for {symbol}: {signal['signal']} (strength: {signal['strength']:.2f}, confidence: {signal['confidence']:.2f})")
                
                # Store signal
                self.signals[symbol] = signal
                
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {e}")
                # Use neutral signal as fallback
                self.signals[symbol] = signal
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {
                "function": "generate_signals",
                "symbol": symbol
            })
            
            self.logger.error(f"Error generating signals for {symbol}: {str(e)}")

async def execute_trades(self):
    """Execute trades based on generated signals."""
    # Skip trade execution if in monitor-only mode
    if self.mode_manager.get_current_mode() == TradingMode.MONITOR_ONLY:
        self.logger.info("Monitor-only mode active, skipping trade execution")
        return
        
    self.logger.info("Executing trades...")
    
    # Get symbols to monitor
    symbols = self.config.get("symbols", ["BTC", "ETH", "SOL"])
    
    # Check if we have signals
    if not hasattr(self, 'signals') or not self.signals:
        self.logger.warning("No signals available, skipping trade execution")
        return
    
    for symbol in symbols:
        try:
            # Skip if no signal for this symbol
            if symbol not in self.signals:
                self.logger.info(f"No signal for {symbol}, skipping trade execution")
                continue
            
            # Get signal
            signal = self.signals[symbol]
            
            # Skip if neutral signal
            if signal["signal"] == "NEUTRAL":
                self.logger.info(f"Neutral signal for {symbol}, no trade execution needed")
                continue
            
            # Get current position
            current_position = self.positions.get(symbol, {"size": 0})
            current_position_size = current_position.get("size", 0)
            
            # Get target position size
            target_position_size = signal.get("target_position_size", 0)
            
            # Calculate position delta
            position_delta = target_position_size - current_position_size
            
            # Skip if position delta is too small
            min_trade_size = self.config.get("min_trade_size", 0.001)
            if abs(position_delta) < min_trade_size:
                self.logger.info(f"Position delta for {symbol} too small ({position_delta}), skipping trade")
                continue
            
            # Determine if this is a new position, position increase, or position decrease
            is_new_position = current_position_size == 0
            is_position_increase = (current_position_size > 0 and position_delta > 0) or (current_position_size < 0 and position_delta < 0)
            is_position_decrease = (current_position_size > 0 and position_delta < 0) or (current_position_size < 0 and position_delta > 0)
            is_position_flip = (current_position_size > 0 and target_position_size < 0) or (current_position_size < 0 and target_position_size > 0)
            
            # Get current price
            current_price = self.market_data.get(symbol, {}).get("price", 0)
            if current_price == 0:
                self.logger.warning(f"Invalid price for {symbol}, skipping trade")
                continue
            
            # Execute trade based on position change type
            if is_new_position or is_position_increase:
                # New position or increase existing position
                is_buy = position_delta > 0
                size = abs(position_delta)
                
                self.logger.info(f"Opening {'long' if is_buy else 'short'} position for {symbol}: {size} units at {current_price}")
                
                # Only execute real orders if in live trading mode
                if self.mode_manager.is_real_trading():
                    # Place market order
                    order_result = await self.exchange.place_order(
                        symbol=symbol,
                        size=size,
                        price=None,  # Market order
                        is_buy=is_buy,
                        reduce_only=False
                    )
                    
                    if "error" in order_result:
                        self.logger.error(f"Error placing order for {symbol}: {order_result['error']}")
                    else:
                        self.logger.info(f"Order placed successfully for {symbol}")
                        
                        # Update last trade time
                        self.last_trade_time = time.time()
                        
                        # Set stop loss and take profit if available
                        if "stop_loss" in signal and signal["stop_loss"] > 0:
                            await self.exchange.place_stop_loss(
                                symbol=symbol,
                                size=size,
                                stop_price=signal["stop_loss"],
                                is_buy=not is_buy  # Opposite direction for stop loss
                            )
                            
                        if "take_profit" in signal and signal["take_profit"] > 0:
                            await self.exchange.place_take_profit(
                                symbol=symbol,
                                size=size,
                                take_profit_price=signal["take_profit"],
                                is_buy=not is_buy  # Opposite direction for take profit
                            )
                else:
                    self.logger.info(f"Simulating {'buy' if is_buy else 'sell'} order for {symbol}: {size} units at {current_price}")
                    
                    # Simulate order execution
                    # In a real implementation, this would update a simulated portfolio
                    
                    # Update last trade time
                    self.last_trade_time = time.time()
            
            elif is_position_decrease and not is_position_flip:
                # Decrease existing position
                is_buy = position_delta > 0
                size = abs(position_delta)
                
                self.logger.info(f"Reducing {'long' if current_position_size > 0 else 'short'} position for {symbol}: {size} units at {current_price}")
                
                # Only execute real orders if in live trading mode
                if self.mode_manager.is_real_trading():
                    # Place market order with reduce_only=True
                    order_result = await self.exchange.place_order(
                        symbol=symbol,
                        size=size,
                        price=None,  # Market order
                        is_buy=is_buy,
                        reduce_only=True
                    )
                    
                    if "error" in order_result:
                        self.logger.error(f"Error placing order for {symbol}: {order_result['error']}")
                    else:
                        self.logger.info(f"Order placed successfully for {symbol}")
                        
                        # Update last trade time
                        self.last_trade_time = time.time()
                else:
                    self.logger.info(f"Simulating {'buy' if is_buy else 'sell'} order for {symbol}: {size} units at {current_price}")
                    
                    # Simulate order execution
                    # In a real implementation, this would update a simulated portfolio
                    
                    # Update last trade time
                    self.last_trade_time = time.time()
            
            elif is_position_flip:
                # Close existing position and open new one in opposite direction
                
                # First, close existing position
                is_buy = current_position_size < 0  # Buy to close short, sell to close long
                size = abs(current_position_size)
                
                self.logger.info(f"Closing {'long' if current_position_size > 0 else 'short'} position for {symbol}: {size} units at {current_price}")
                
                # Only execute real orders if in live trading mode
                if self.mode_manager.is_real_trading():
                    # Place market order to close position
                    close_result = await self.exchange.place_order(
                        symbol=symbol,
                        size=size,
                        price=None,  # Market order
                        is_buy=is_buy,
                        reduce_only=True
                    )
                    
                    if "error" in close_result:
                        self.logger.error(f"Error closing position for {symbol}: {close_result['error']}")
                        continue  # Skip opening new position if closing failed
                    
                    # Then, open new position
                    is_buy = target_position_size > 0
                    size = abs(target_position_size)
                    
                    self.logger.info(f"Opening {'long' if is_buy else 'short'} position for {symbol}: {size} units at {current_price}")
                    
                    # Place market order for new position
                    open_result = await self.exchange.place_order(
                        symbol=symbol,
                        size=size,
                        price=None,  # Market order
                        is_buy=is_buy,
                        reduce_only=False
                    )
                    
                    if "error" in open_result:
                        self.logger.error(f"Error opening new position for {symbol}: {open_result['error']}")
                    else:
                        self.logger.info(f"New position opened successfully for {symbol}")
                        
                        # Update last trade time
                        self.last_trade_time = time.time()
                        
                        # Set stop loss and take profit if available
                        if "stop_loss" in signal and signal["stop_loss"] > 0:
                            await self.exchange.place_stop_loss(
                                symbol=symbol,
                                size=size,
                                stop_price=signal["stop_loss"],
                                is_buy=not is_buy  # Opposite direction for stop loss
                            )
                            
                        if "take_profit" in signal and signal["take_profit"] > 0:
                            await self.exchange.place_take_profit(
                                symbol=symbol,
                                size=size,
                                take_profit_price=signal["take_profit"],
                                is_buy=not is_buy  # Opposite direction for take profit
                            )
                else:
                    self.logger.info(f"Simulating position flip for {symbol} from {current_position_size} to {target_position_size}")
                    
                    # Simulate order execution
                    # In a real implementation, this would update a simulated portfolio
                    
                    # Update last trade time
                    self.last_trade_time = time.time()
            
        except Exception as e:
            error_info = self.error_handler.handle_error(e, {
                "function": "execute_trades",
                "symbol": symbol
            })
            
            self.logger.error(f"Error executing trades for {symbol}: {str(e)}")
