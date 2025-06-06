"""
Strategy Manager for Hyperliquid Trading Bot
Integrates with connection manager to ensure trading strategies always have a valid connection
"""

import os
import sys
import time
import threading
import logging
from typing import Dict, Any, Optional, List, Type

from utils.logger import get_logger
from utils.config_manager_fixed import ConfigManager
from core.connection_manager_enhanced import EnhancedConnectionManager
from strategies.base_strategy import BaseStrategy
from strategies.bb_rsi_adx import BB_RSI_ADX
from strategies.hull_suite import HullSuite
from risk_management.risk_manager import RiskManager

logger = get_logger(__name__)


class StrategyManager:
    """
    Strategy Manager for Hyperliquid Trading Bot
    Integrates with connection manager to ensure trading strategies always have a valid connection
    """
    
    def __init__(self, connection_manager=None, config_manager=None):
        """
        Initialize the strategy manager
        
        Args:
            connection_manager: Connection manager instance
            config_manager: Configuration manager instance
        """
        self.connection_manager = connection_manager or EnhancedConnectionManager()
        self.config_manager = config_manager or ConfigManager()
        self.risk_manager = RiskManager()
        
        # Initialize strategies
        self.strategies = {}
        self.active_strategies = []
        
        # Register available strategies
        self._register_strategies()
        
        # Load configuration
        self._load_config()
        
        # Initialize strategy execution
        self.running = False
        self.execution_thread = None
        
        logger.info("Strategy Manager initialized")
    
    def _register_strategies(self):
        """Register available strategies"""
        try:
            # Register BB_RSI_ADX strategy
            self.strategies["BB_RSI_ADX"] = BB_RSI_ADX
            
            # Register Hull Suite strategy
            self.strategies["Hull_Suite"] = HullSuite
            
            logger.info(f"Registered {len(self.strategies)} strategies")
        except Exception as e:
            logger.error(f"Failed to register strategies: {e}")
    
    def _load_config(self):
        """Load configuration"""
        try:
            # Get active strategies
            active_strategies = self.config_manager.get('trading.active_strategies', ["BB_RSI_ADX"])
            
            # Initialize active strategies
            for strategy_name in active_strategies:
                if strategy_name in self.strategies:
                    strategy_class = self.strategies[strategy_name]
                    strategy_instance = strategy_class(
                        api=self.connection_manager.api,
                        risk_manager=self.risk_manager
                    )
                    self.active_strategies.append(strategy_instance)
            
            logger.info(f"Loaded {len(self.active_strategies)} active strategies")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    def start(self):
        """Start strategy execution"""
        try:
            if self.running:
                logger.warning("Strategy execution is already running")
                return False
            
            # Check if we have active strategies
            if not self.active_strategies:
                logger.warning("No active strategies to execute")
                return False
            
            # Check if we have a valid connection
            if not self.connection_manager.is_connected:
                logger.warning("No valid connection to execute strategies")
                if not self.connection_manager.ensure_connection():
                    logger.error("Failed to establish connection")
                    return False
            
            # Start execution thread
            self.running = True
            self.execution_thread = threading.Thread(target=self._execution_loop)
            self.execution_thread.daemon = True
            self.execution_thread.start()
            
            logger.info("Strategy execution started")
            return True
        except Exception as e:
            logger.error(f"Failed to start strategy execution: {e}")
            return False
    
    def stop(self):
        """Stop strategy execution"""
        try:
            if not self.running:
                logger.warning("Strategy execution is not running")
                return False
            
            # Stop execution thread
            self.running = False
            
            # Wait for thread to terminate
            if self.execution_thread and self.execution_thread.is_alive():
                self.execution_thread.join(timeout=5.0)
            
            logger.info("Strategy execution stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop strategy execution: {e}")
            return False
    
    def _execution_loop(self):
        """Strategy execution loop"""
        try:
            logger.info("Strategy execution loop started")
            
            while self.running:
                # Check connection health
                if not self.connection_manager.check_connection_health():
                    logger.warning("Connection is not healthy, attempting to reconnect")
                    if not self.connection_manager.ensure_connection():
                        logger.error("Failed to reconnect, pausing strategy execution")
                        time.sleep(30)  # Wait before retrying
                        continue
                
                # Execute strategies
                for strategy in self.active_strategies:
                    try:
                        # Update strategy with latest API instance
                        strategy.api = self.connection_manager.api
                        
                        # Execute strategy
                        strategy.execute()
                    except Exception as e:
                        logger.error(f"Failed to execute strategy {strategy.name}: {e}")
                
                # Sleep before next iteration
                time.sleep(10)
        except Exception as e:
            logger.error(f"Strategy execution loop failed: {e}")
        finally:
            logger.info("Strategy execution loop ended")
            self.running = False
    
    def add_strategy(self, strategy_name):
        """
        Add a strategy to active strategies
        
        Args:
            strategy_name: Name of the strategy to add
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if strategy exists
            if strategy_name not in self.strategies:
                logger.error(f"Strategy {strategy_name} not found")
                return False
            
            # Check if strategy is already active
            for strategy in self.active_strategies:
                if strategy.name == strategy_name:
                    logger.warning(f"Strategy {strategy_name} is already active")
                    return False
            
            # Initialize strategy
            strategy_class = self.strategies[strategy_name]
            strategy_instance = strategy_class(
                api=self.connection_manager.api,
                risk_manager=self.risk_manager
            )
            
            # Add to active strategies
            self.active_strategies.append(strategy_instance)
            
            # Update configuration
            active_strategies = [s.name for s in self.active_strategies]
            self.config_manager.set('trading.active_strategies', active_strategies)
            self.config_manager.save_config()
            
            logger.info(f"Added strategy {strategy_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add strategy {strategy_name}: {e}")
            return False
    
    def remove_strategy(self, strategy_name):
        """
        Remove a strategy from active strategies
        
        Args:
            strategy_name: Name of the strategy to remove
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Find strategy
            for i, strategy in enumerate(self.active_strategies):
                if strategy.name == strategy_name:
                    # Remove from active strategies
                    self.active_strategies.pop(i)
                    
                    # Update configuration
                    active_strategies = [s.name for s in self.active_strategies]
                    self.config_manager.set('trading.active_strategies', active_strategies)
                    self.config_manager.save_config()
                    
                    logger.info(f"Removed strategy {strategy_name}")
                    return True
            
            logger.warning(f"Strategy {strategy_name} not found in active strategies")
            return False
        except Exception as e:
            logger.error(f"Failed to remove strategy {strategy_name}: {e}")
            return False
    
    def get_active_strategies(self):
        """
        Get active strategies
        
        Returns:
            list: List of active strategy names
        """
        try:
            return [strategy.name for strategy in self.active_strategies]
        except Exception as e:
            logger.error(f"Failed to get active strategies: {e}")
            return []
    
    def get_available_strategies(self):
        """
        Get available strategies
        
        Returns:
            list: List of available strategy names
        """
        try:
            return list(self.strategies.keys())
        except Exception as e:
            logger.error(f"Failed to get available strategies: {e}")
            return []
    
    def get_strategy_status(self, strategy_name):
        """
        Get strategy status
        
        Args:
            strategy_name: Name of the strategy
        
        Returns:
            dict: Strategy status
        """
        try:
            # Find strategy
            for strategy in self.active_strategies:
                if strategy.name == strategy_name:
                    return {
                        "name": strategy.name,
                        "active": True,
                        "positions": strategy.get_positions(),
                        "orders": strategy.get_orders(),
                        "performance": strategy.get_performance()
                    }
            
            # Strategy not active
            if strategy_name in self.strategies:
                return {
                    "name": strategy_name,
                    "active": False,
                    "positions": [],
                    "orders": [],
                    "performance": {}
                }
            
            logger.warning(f"Strategy {strategy_name} not found")
            return None
        except Exception as e:
            logger.error(f"Failed to get strategy status for {strategy_name}: {e}")
            return None
    
    def get_all_strategy_status(self):
        """
        Get status of all strategies
        
        Returns:
            list: List of strategy status dictionaries
        """
        try:
            result = []
            
            # Get status of active strategies
            for strategy in self.active_strategies:
                result.append({
                    "name": strategy.name,
                    "active": True,
                    "positions": strategy.get_positions(),
                    "orders": strategy.get_orders(),
                    "performance": strategy.get_performance()
                })
            
            # Get status of inactive strategies
            for strategy_name in self.strategies:
                if strategy_name not in [s.name for s in self.active_strategies]:
                    result.append({
                        "name": strategy_name,
                        "active": False,
                        "positions": [],
                        "orders": [],
                        "performance": {}
                    })
            
            return result
        except Exception as e:
            logger.error(f"Failed to get all strategy status: {e}")
            return []
    
    def update_strategy_parameters(self, strategy_name, parameters):
        """
        Update strategy parameters
        
        Args:
            strategy_name: Name of the strategy
            parameters: Dictionary of parameters to update
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Find strategy
            for strategy in self.active_strategies:
                if strategy.name == strategy_name:
                    # Update parameters
                    strategy.update_parameters(parameters)
                    
                    logger.info(f"Updated parameters for strategy {strategy_name}")
                    return True
            
            logger.warning(f"Strategy {strategy_name} not found in active strategies")
            return False
        except Exception as e:
            logger.error(f"Failed to update parameters for strategy {strategy_name}: {e}")
            return False
    
    def get_strategy_parameters(self, strategy_name):
        """
        Get strategy parameters
        
        Args:
            strategy_name: Name of the strategy
        
        Returns:
            dict: Strategy parameters
        """
        try:
            # Find strategy
            for strategy in self.active_strategies:
                if strategy.name == strategy_name:
                    return strategy.get_parameters()
            
            # Strategy not active, get default parameters
            if strategy_name in self.strategies:
                strategy_class = self.strategies[strategy_name]
                return strategy_class.get_default_parameters()
            
            logger.warning(f"Strategy {strategy_name} not found")
            return None
        except Exception as e:
            logger.error(f"Failed to get parameters for strategy {strategy_name}: {e}")
            return None
    
    def reset_strategy(self, strategy_name):
        """
        Reset strategy
        
        Args:
            strategy_name: Name of the strategy
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Find strategy
            for strategy in self.active_strategies:
                if strategy.name == strategy_name:
                    # Reset strategy
                    strategy.reset()
                    
                    logger.info(f"Reset strategy {strategy_name}")
                    return True
            
            logger.warning(f"Strategy {strategy_name} not found in active strategies")
            return False
        except Exception as e:
            logger.error(f"Failed to reset strategy {strategy_name}: {e}")
            return False
    
    def reset_all_strategies(self):
        """
        Reset all strategies
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Reset all strategies
            for strategy in self.active_strategies:
                strategy.reset()
            
            logger.info("Reset all strategies")
            return True
        except Exception as e:
            logger.error(f"Failed to reset all strategies: {e}")
            return False

