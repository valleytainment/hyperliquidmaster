# Step-by-Step Guide for the Enhanced Hyperliquid Trading Bot

This comprehensive guide will walk you through the process of setting up, testing, training, and running the Enhanced Hyperliquid Trading Bot with the Master Omni Overlord Strategy.

## Table of Contents
1. [Installation and Setup](#installation-and-setup)
2. [Configuration](#configuration)
3. [Testing the Bot](#testing-the-bot)
4. [Training the Bot](#training-the-bot)
5. [Running the Bot](#running-the-bot)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Configuration](#advanced-configuration)

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for version control)

### Step 1: Install the Bot
1. Unzip the `enhanced_hyperliquid_bot.zip` file to your desired location:
   ```bash
   unzip enhanced_hyperliquid_bot.zip -d /path/to/destination
   ```

2. Navigate to the bot directory:
   ```bash
   cd /path/to/destination/enhanced_bot
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Step 1: Basic Configuration
1. Copy your existing `config.json` file to the bot directory, or use the provided template:
   ```bash
   cp /path/to/your/config.json ./config.json
   ```

2. Run the configuration compatibility tool to ensure your config is compatible with the enhanced bot:
   ```bash
   python config_compatibility.py
   ```

3. The tool will automatically update your configuration file while maintaining compatibility with your existing Hyperliquid SDK.

### Step 2: Strategy Configuration
1. Open the `config.json` file and locate the `strategies` section.
2. Configure the Master Omni Overlord Strategy parameters:
   ```json
   "strategies": {
     "master_omni_overlord": {
       "enabled": true,
       "volatility_lookback": 15,
       "trend_lookback": 30,
       "regime_threshold": 0.65,
       "signal_threshold": 0.7,
       "risk_per_trade": 0.015
     }
   }
   ```

3. Configure the sub-strategies if you want to modify their default parameters:
   ```json
   "triple_confluence": {
     "enabled": true,
     "funding_threshold": 0.0001,
     "order_imbalance_threshold": 1.3
   },
   "oracle_update": {
     "enabled": true,
     "min_price_deviation": 0.0025,
     "max_price_deviation": 0.01
   }
   ```

### Step 3: Risk Management Configuration
1. Configure the risk management parameters:
   ```json
   "risk_management": {
     "max_position_size_percent": 0.1,
     "max_drawdown_percent": 0.07,
     "circuit_breaker_losses": 3,
     "circuit_breaker_timeout_hours": 24,
     "use_dynamic_position_sizing": true,
     "use_tiered_stop_loss": true
   }
   ```

## Testing the Bot

### Step 1: Run Integration Tests
1. Run the integration test to ensure all components are working correctly:
   ```bash
   python integration_test.py
   ```

2. The test will verify:
   - Hyperliquid SDK connectivity
   - Strategy initialization
   - Signal generation
   - Risk management calculations
   - Order execution simulation

3. Review the test results and fix any issues before proceeding.

### Step 2: Run Backtesting
1. Run a backtest to evaluate the strategy performance:
   ```bash
   python backtest_trainer.py --mode backtest --symbol BTC-USD-PERP --days 30
   ```

2. Review the backtest results in the `backtest_results` directory:
   ```bash
   ls -la backtest_results/
   ```

3. Key metrics to review:
   - Win rate
   - Profit factor
   - Maximum drawdown
   - Sharpe ratio

## Training the Bot

### Step 1: Collect Training Data
1. Run the data collector to gather market data:
   ```bash
   python data_collector.py --symbols BTC-USD-PERP,ETH-USD-PERP,SOL-USD-PERP --days 60
   ```

2. Verify the data collection:
   ```bash
   ls -la data/
   ```

### Step 2: Run Parameter Optimization
1. Run the training script to optimize strategy parameters:
   ```bash
   python run_master_strategy_training.py --symbols BTC-USD-PERP --epochs 5
   ```

2. For more extensive training:
   ```bash
   python run_master_strategy_training.py --symbols BTC-USD-PERP,ETH-USD-PERP,SOL-USD-PERP --epochs 10
   ```

3. Review the training results:
   ```bash
   cat training_results/master_strategy_results.md
   ```

### Step 3: Apply Optimized Parameters
1. The training process will automatically update your `config.json` with the optimized parameters.
2. Verify the changes:
   ```bash
   cat config.json
   ```

## Running the Bot

### Step 1: Dry Run Mode
1. Run the bot in dry run mode (no real trades):
   ```bash
   python main.py --dry-run
   ```

2. Monitor the output for at least 1 hour to ensure everything is working correctly.
3. Check the logs for any warnings or errors:
   ```bash
   tail -f logs/trading_bot.log
   ```

### Step 2: Live Trading
1. Once you're confident in the bot's performance, run it in live mode:
   ```bash
   python main.py
   ```

2. For running as a background process:
   ```bash
   nohup python main.py > trading.out 2>&1 &
   ```

3. To stop the bot:
   ```bash
   pkill -f "python main.py"
   ```

## Monitoring and Maintenance

### Daily Monitoring
1. Check the bot's performance:
   ```bash
   python performance_report.py
   ```

2. Review the logs:
   ```bash
   tail -f logs/trading_bot.log
   ```

3. Check for any circuit breaker activations:
   ```bash
   grep "Circuit breaker activated" logs/trading_bot.log
   ```

### Weekly Maintenance
1. Update market data:
   ```bash
   python data_collector.py --update
   ```

2. Run a quick backtest to ensure strategy performance remains strong:
   ```bash
   python backtest_trainer.py --mode backtest --symbol BTC-USD-PERP --days 7
   ```

3. Check for any required updates to the bot:
   ```bash
   python check_updates.py
   ```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Bot fails to connect to Hyperliquid
1. Check your API keys in `config.json`
2. Verify network connectivity
3. Ensure you're not IP-restricted

Solution:
```bash
python connectivity_test.py
```

#### Issue: No trading signals generated
1. Check if market data is being properly collected
2. Verify strategy parameters
3. Check for missing order book data

Solution:
```bash
python data_validator.py
```

#### Issue: Unexpected trading behavior
1. Check risk management settings
2. Review recent market conditions
3. Verify strategy weights

Solution:
```bash
python strategy_analyzer.py --last-trades 10
```

## Advanced Configuration

### Customizing the Master Omni Overlord Strategy
1. Edit the strategy weights:
   ```json
   "strategy_weights": {
     "trending": {
       "triple_confluence": 0.7,
       "oracle_update": 0.3
     },
     "mean_reverting": {
       "triple_confluence": 0.3,
       "oracle_update": 0.7
     }
   }
   ```

2. Adjust the adaptive parameters:
   ```json
   "adaptive_parameters": {
     "signal_threshold": {
       "min": 0.5,
       "max": 0.8,
       "default": 0.7
     },
     "trend_filter_strength": {
       "min": 0.6,
       "max": 0.9,
       "default": 0.8
     }
   }
   ```

### Integrating with External Data Sources
1. Configure sentiment analysis:
   ```json
   "sentiment_analysis": {
     "enabled": true,
     "sources": ["twitter", "reddit", "news"],
     "update_interval_minutes": 60,
     "influence_weight": 0.2
   }
   ```

2. Configure on-chain data:
   ```json
   "on_chain_data": {
     "enabled": true,
     "metrics": ["exchange_flows", "whale_transactions", "gas_prices"],
     "update_interval_minutes": 30,
     "influence_weight": 0.15
   }
   ```

### Performance Tuning
1. Adjust the execution parameters:
   ```json
   "execution": {
     "use_scale_orders": true,
     "slippage_tolerance": 0.001,
     "retry_attempts": 3,
     "retry_delay_seconds": 5
   }
   ```

2. Configure the circuit breaker:
   ```json
   "circuit_breaker": {
     "consecutive_losses": 3,
     "drawdown_percent": 0.07,
     "timeout_hours": 24,
     "partial_reset": true
   }
   ```

---

This guide covers the essential steps for setting up, testing, training, and running the Enhanced Hyperliquid Trading Bot. For more detailed information, refer to the individual module documentation in the `docs` directory.

If you encounter any issues not covered in this guide, please check the troubleshooting section or contact support.

Happy trading!
