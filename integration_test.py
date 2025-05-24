"""
Integration Test Module

This module provides comprehensive testing capabilities for the enhanced trading bot,
ensuring all components work together correctly.
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any

# Import core components
from core.hyperliquid_adapter import HyperliquidExchangeAdapter
from core.error_handler import ErrorHandler
from sentiment.llm_analyzer import LLMSentimentAnalyzer
from strategies.triple_confluence import TripleConfluenceStrategy
from strategies.oracle_update import OracleUpdateStrategy
from config_compatibility import ConfigManager

class IntegrationTester:
    """
    Integration tester for the enhanced trading bot.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the integration tester.
        
        Args:
            config_path: Path to the configuration file
        """
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info("Initializing Integration Tester...")
        
        # Load configuration
        self.config_manager = ConfigManager(config_path, self.logger)
        self.config = self.config_manager.get_config()
        
        # Initialize error handler
        self.error_handler = ErrorHandler(self.logger)
        
        # Initialize components
        self.exchange = None
        self.sentiment_analyzer = None
        self.triple_confluence_strategy = None
        self.oracle_update_strategy = None
        
        # Test results
        self.test_results = {
            "exchange_adapter": {"status": "pending", "details": {}},
            "sentiment_analyzer": {"status": "pending", "details": {}},
            "triple_confluence_strategy": {"status": "pending", "details": {}},
            "oracle_update_strategy": {"status": "pending", "details": {}},
            "config_compatibility": {"status": "pending", "details": {}},
            "end_to_end": {"status": "pending", "details": {}}
        }
        
    def _setup_logger(self) -> logging.Logger:
        """
        Set up the logger.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger("IntegrationTester")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("integration_test.log", mode="a")
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
        
    async def run_all_tests(self):
        """Run all integration tests."""
        self.logger.info("Starting integration tests...")
        
        # Test configuration compatibility
        await self.test_config_compatibility()
        
        # Test exchange adapter
        await self.test_exchange_adapter()
        
        # Test sentiment analyzer
        await self.test_sentiment_analyzer()
        
        # Test strategies
        await self.test_strategies()
        
        # Test end-to-end flow
        await self.test_end_to_end()
        
        # Save test results
        self.save_test_results()
        
        self.logger.info("Integration tests completed.")
        
    async def test_config_compatibility(self):
        """Test configuration compatibility."""
        self.logger.info("Testing configuration compatibility...")
        
        try:
            # Test original config format conversion
            original_format = self.config_manager.get_original_config_format()
            
            # Validate configuration
            validation_results = self.config_manager.validate_config()
            
            # Check if validation passed
            if validation_results["valid"]:
                self.test_results["config_compatibility"]["status"] = "passed"
                self.test_results["config_compatibility"]["details"] = {
                    "warnings": validation_results["warnings"],
                    "original_format_keys": list(original_format.keys())
                }
                self.logger.info("Configuration compatibility test passed with warnings: " + 
                               str(len(validation_results["warnings"])))
            else:
                self.test_results["config_compatibility"]["status"] = "failed"
                self.test_results["config_compatibility"]["details"] = {
                    "errors": validation_results["errors"],
                    "warnings": validation_results["warnings"]
                }
                self.logger.error("Configuration compatibility test failed: " + 
                                str(validation_results["errors"]))
        except Exception as e:
            self.test_results["config_compatibility"]["status"] = "error"
            self.test_results["config_compatibility"]["details"] = {"error": str(e)}
            self.logger.exception("Error in configuration compatibility test")
            
    async def test_exchange_adapter(self):
        """Test exchange adapter."""
        self.logger.info("Testing exchange adapter...")
        
        try:
            # Initialize exchange adapter
            self.exchange = HyperliquidExchangeAdapter(
                self.config,
                self.logger,
                self.error_handler
            )
            
            # Initialize exchange connection
            connection_result = await self.exchange.initialize()
            
            if not connection_result:
                self.test_results["exchange_adapter"]["status"] = "failed"
                self.test_results["exchange_adapter"]["details"] = {
                    "error": "Failed to initialize exchange connection"
                }
                self.logger.error("Exchange adapter test failed: connection initialization failed")
                return
                
            # Test market data retrieval
            symbols = self.config.get("symbols", ["BTC-USD-PERP"])
            market_data = await self.exchange.fetch_market_data(symbols)
            
            if not market_data:
                self.test_results["exchange_adapter"]["status"] = "failed"
                self.test_results["exchange_adapter"]["details"] = {
                    "error": "Failed to retrieve market data"
                }
                self.logger.error("Exchange adapter test failed: market data retrieval failed")
                return
                
            # Test order book retrieval
            order_book = await self.exchange.fetch_order_book(symbols[0])
            
            # Test user positions retrieval
            positions = await self.exchange.get_user_positions()
            
            # Test account equity retrieval
            equity = await self.exchange.get_account_equity()
            
            self.test_results["exchange_adapter"]["status"] = "passed"
            self.test_results["exchange_adapter"]["details"] = {
                "market_data_symbols": list(market_data.keys()),
                "order_book_levels": len(order_book["bids"]) if order_book else 0,
                "positions_count": len(positions),
                "account_equity": equity
            }
            
            self.logger.info("Exchange adapter test passed")
            
        except Exception as e:
            self.test_results["exchange_adapter"]["status"] = "error"
            self.test_results["exchange_adapter"]["details"] = {"error": str(e)}
            self.logger.exception("Error in exchange adapter test")
            
    async def test_sentiment_analyzer(self):
        """Test sentiment analyzer."""
        self.logger.info("Testing sentiment analyzer...")
        
        if not self.config.get("use_sentiment_analysis", True):
            self.test_results["sentiment_analyzer"]["status"] = "skipped"
            self.test_results["sentiment_analyzer"]["details"] = {
                "reason": "Sentiment analysis disabled in configuration"
            }
            self.logger.info("Sentiment analyzer test skipped: disabled in configuration")
            return
            
        try:
            # Initialize sentiment analyzer
            self.sentiment_analyzer = LLMSentimentAnalyzer(
                self.config,
                self.logger
            )
            
            # Test with mock data
            mock_news = [
                {
                    "title": "Bitcoin Surges Past $50,000 as Institutional Adoption Grows",
                    "content": "Bitcoin has surged past $50,000 as institutional adoption continues to grow.",
                    "published_at": "2025-05-23T12:00:00Z"
                }
            ]
            
            mock_posts = [
                {
                    "text": "Just bought more $BTC, feeling bullish about the market!",
                    "platform": "twitter",
                    "timestamp": "2025-05-23T12:30:00Z"
                }
            ]
            
            # Test narrative detection
            narratives = await self.sentiment_analyzer.detect_market_narratives(
                mock_news, mock_posts
            )
            
            if narratives:
                self.test_results["sentiment_analyzer"]["status"] = "passed"
                self.test_results["sentiment_analyzer"]["details"] = {
                    "narratives_detected": len(narratives.get("narratives", [])),
                    "market_regime": narratives.get("market_regime", "unknown")
                }
                self.logger.info("Sentiment analyzer test passed")
            else:
                self.test_results["sentiment_analyzer"]["status"] = "failed"
                self.test_results["sentiment_analyzer"]["details"] = {
                    "error": "Failed to detect market narratives"
                }
                self.logger.error("Sentiment analyzer test failed: narrative detection failed")
                
        except Exception as e:
            self.test_results["sentiment_analyzer"]["status"] = "error"
            self.test_results["sentiment_analyzer"]["details"] = {"error": str(e)}
            self.logger.exception("Error in sentiment analyzer test")
            
    async def test_strategies(self):
        """Test trading strategies."""
        self.logger.info("Testing trading strategies...")
        
        try:
            # Initialize strategies
            self.triple_confluence_strategy = TripleConfluenceStrategy(
                self.config,
                self.logger
            )
            
            self.oracle_update_strategy = OracleUpdateStrategy(
                self.config,
                self.logger
            )
            
            # Test with mock data
            symbol = self.config.get("symbols", ["BTC-USD-PERP"])[0]
            
            # Mock data for Triple Confluence strategy
            price_data = [50000 + i * 100 for i in range(100)]
            volume_data = [100 + i for i in range(100)]
            funding_rate = -0.0001  # Negative funding rate
            
            # Create mock order book
            mock_order_book = {
                "bids": [[50000 - i * 10, 10 + i] for i in range(10)],
                "asks": [[50000 + i * 10, 5 + i] for i in range(10)]
            }
            
            # Update Triple Confluence strategy data
            for i in range(100):
                self.triple_confluence_strategy.update_data(
                    symbol,
                    price_data[i],
                    volume_data[i],
                    funding_rate,
                    mock_order_book
                )
                
            # Analyze with Triple Confluence strategy
            tc_result = self.triple_confluence_strategy.analyze(symbol)
            
            # Mock data for Oracle Update strategy
            market_price = 50000
            oracle_price = 50100  # Oracle price higher than market price
            
            # Update Oracle Update strategy data
            self.oracle_update_strategy.update_data(
                symbol,
                market_price,
                oracle_price
            )
            
            # Analyze with Oracle Update strategy
            ou_result = self.oracle_update_strategy.analyze(symbol)
            
            # Check strategy results
            if tc_result and ou_result:
                self.test_results["triple_confluence_strategy"]["status"] = "passed"
                self.test_results["triple_confluence_strategy"]["details"] = {
                    "signal": tc_result.get("signal"),
                    "confidence": tc_result.get("confidence"),
                    "reason": tc_result.get("reason")
                }
                
                self.test_results["oracle_update_strategy"]["status"] = "passed"
                self.test_results["oracle_update_strategy"]["details"] = {
                    "signal": ou_result.get("signal"),
                    "confidence": ou_result.get("confidence"),
                    "reason": ou_result.get("reason")
                }
                
                self.logger.info("Strategy tests passed")
            else:
                if not tc_result:
                    self.test_results["triple_confluence_strategy"]["status"] = "failed"
                    self.test_results["triple_confluence_strategy"]["details"] = {
                        "error": "Failed to analyze with Triple Confluence strategy"
                    }
                    self.logger.error("Triple Confluence strategy test failed")
                    
                if not ou_result:
                    self.test_results["oracle_update_strategy"]["status"] = "failed"
                    self.test_results["oracle_update_strategy"]["details"] = {
                        "error": "Failed to analyze with Oracle Update strategy"
                    }
                    self.logger.error("Oracle Update strategy test failed")
                    
        except Exception as e:
            self.test_results["triple_confluence_strategy"]["status"] = "error"
            self.test_results["triple_confluence_strategy"]["details"] = {"error": str(e)}
            self.test_results["oracle_update_strategy"]["status"] = "error"
            self.test_results["oracle_update_strategy"]["details"] = {"error": str(e)}
            self.logger.exception("Error in strategy tests")
            
    async def test_end_to_end(self):
        """Test end-to-end flow."""
        self.logger.info("Testing end-to-end flow...")
        
        try:
            # Check if all components are initialized
            if not self.exchange:
                self.test_results["end_to_end"]["status"] = "skipped"
                self.test_results["end_to_end"]["details"] = {
                    "reason": "Exchange adapter not initialized"
                }
                self.logger.warning("End-to-end test skipped: exchange adapter not initialized")
                return
                
            # Get market data
            symbols = self.config.get("symbols", ["BTC-USD-PERP"])
            market_data = await self.exchange.fetch_market_data(symbols)
            
            if not market_data:
                self.test_results["end_to_end"]["status"] = "failed"
                self.test_results["end_to_end"]["details"] = {
                    "error": "Failed to retrieve market data"
                }
                self.logger.error("End-to-end test failed: market data retrieval failed")
                return
                
            # Get order book
            symbol = symbols[0]
            order_book = await self.exchange.fetch_order_book(symbol)
            
            if not order_book:
                self.test_results["end_to_end"]["status"] = "failed"
                self.test_results["end_to_end"]["details"] = {
                    "error": "Failed to retrieve order book"
                }
                self.logger.error("End-to-end test failed: order book retrieval failed")
                return
                
            # Get oracle price
            oracle_price = await self.exchange.fetch_oracle_price(symbol)
            
            # Update strategy data
            if self.triple_confluence_strategy and self.oracle_update_strategy:
                price = market_data[symbol]["price"]
                volume = market_data[symbol]["volume"]
                funding_rate = market_data[symbol]["funding_rate"]
                
                self.triple_confluence_strategy.update_data(
                    symbol,
                    price,
                    volume,
                    funding_rate,
                    order_book
                )
                
                if oracle_price:
                    self.oracle_update_strategy.update_data(
                        symbol,
                        price,
                        oracle_price
                    )
                    
                # Analyze with strategies
                tc_result = self.triple_confluence_strategy.analyze(symbol)
                ou_result = self.oracle_update_strategy.analyze(symbol)
                
                # Get sentiment data if available
                sentiment_data = {}
                if self.sentiment_analyzer:
                    # Mock data for testing
                    mock_news = []
                    mock_posts = []
                    
                    sentiment_data = await self.sentiment_analyzer.detect_market_narratives(
                        mock_news, mock_posts
                    )
                    
                # Simulate decision making
                final_signal = "NEUTRAL"
                confidence = 0.0
                
                if tc_result["signal"] != "NEUTRAL" and tc_result["confidence"] > 0.7:
                    final_signal = tc_result["signal"]
                    confidence = tc_result["confidence"]
                elif ou_result["signal"] != "NEUTRAL" and ou_result["confidence"] > 0.7:
                    final_signal = ou_result["signal"]
                    confidence = ou_result["confidence"]
                    
                # Adjust with sentiment if available
                if sentiment_data and self.sentiment_analyzer:
                    # Create a mock signal for testing
                    mock_signal = {
                        "signal": final_signal,
                        "confidence": confidence
                    }
                    
                    adjusted_signal = self.sentiment_analyzer.adjust_trading_signals(
                        mock_signal, sentiment_data
                    )
                    
                    final_signal = adjusted_signal["signal"]
                    confidence = adjusted_signal["confidence"]
                    
                self.test_results["end_to_end"]["status"] = "passed"
                self.test_results["end_to_end"]["details"] = {
                    "final_signal": final_signal,
                    "confidence": confidence,
                    "tc_signal": tc_result["signal"],
                    "ou_signal": ou_result["signal"],
                    "market_price": price,
                    "oracle_price": oracle_price,
                    "funding_rate": funding_rate
                }
                
                self.logger.info(f"End-to-end test passed: final signal {final_signal} with confidence {confidence:.2f}")
            else:
                self.test_results["end_to_end"]["status"] = "skipped"
                self.test_results["end_to_end"]["details"] = {
                    "reason": "Strategies not initialized"
                }
                self.logger.warning("End-to-end test skipped: strategies not initialized")
                
        except Exception as e:
            self.test_results["end_to_end"]["status"] = "error"
            self.test_results["end_to_end"]["details"] = {"error": str(e)}
            self.logger.exception("Error in end-to-end test")
            
    def save_test_results(self):
        """Save test results to file."""
        try:
            with open("integration_test_results.json", "w") as f:
                json.dump(self.test_results, f, indent=2)
                
            self.logger.info("Test results saved to integration_test_results.json")
            
            # Generate summary
            passed = sum(1 for test in self.test_results.values() if test["status"] == "passed")
            failed = sum(1 for test in self.test_results.values() if test["status"] == "failed")
            error = sum(1 for test in self.test_results.values() if test["status"] == "error")
            skipped = sum(1 for test in self.test_results.values() if test["status"] == "skipped")
            
            summary = f"""
            Integration Test Summary
            -----------------------
            Passed: {passed}
            Failed: {failed}
            Error: {error}
            Skipped: {skipped}
            Total: {len(self.test_results)}
            """
            
            with open("integration_test_summary.txt", "w") as f:
                f.write(summary)
                
            self.logger.info(f"Test summary saved to integration_test_summary.txt")
            
        except Exception as e:
            self.logger.error(f"Error saving test results: {str(e)}")

async def main():
    """Main entry point."""
    # Get configuration path from command line arguments
    config_path = "config.json"
    if len(os.sys.argv) > 1:
        config_path = os.sys.argv[1]
        
    # Create and run tester
    tester = IntegrationTester(config_path)
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
