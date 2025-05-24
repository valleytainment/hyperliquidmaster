# Enhanced Hyperliquid Trading Bot with GUI

This package contains the enhanced Hyperliquid trading bot with a fully integrated graphical user interface (GUI). The bot combines robust backend trading algorithms with a user-friendly interface for monitoring and controlling your trading activities.

## Features

### Enhanced Backend
- Robust real market data integration with Hyperliquid exchange
- Historical data accumulation with persistent storage
- Advanced rate limiting protection to prevent API blocks
- Enhanced signal generation with adaptive technical indicators
- Master Omni Overlord Strategy combining multiple trading approaches
- Real-money trading safeguards with circuit breakers

### Comprehensive GUI
- Real-time price charts and visualization
- Technical indicator displays (RSI, MACD, Bollinger Bands, ADX)
- Configuration controls for all trading parameters
- Live monitoring of positions and performance metrics
- Manual trading controls (buy/sell/close position buttons)
- Detailed logging and status updates

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required Python packages (install via `pip install -r requirements.txt`):
  - numpy
  - pandas
  - matplotlib
  - tkinter (usually comes with Python)
  - requests
  - asyncio

### Configuration
1. Edit the `config.json` file to set your Hyperliquid account details and API keys
2. Adjust trading parameters as needed or use the GUI to modify settings

### Running the Bot with GUI
```bash
python gui_main.py
```

This will launch the graphical interface where you can:
- Start/stop the trading bot
- Monitor price charts and indicators
- View and manage positions
- Adjust trading parameters
- Execute manual trades

### Running the Bot without GUI (Headless Mode)
```bash
python main.py
```

## GUI Features Guide

### Main Controls
- **Start Bot**: Begin automated trading with current settings
- **Stop Bot**: Halt all trading activities
- **Symbol Selection**: Choose which cryptocurrency to trade
- **Manual Trading**: Buy, Sell, or Close Position buttons for manual control

### Tabs
1. **Price Chart**: Displays price movement with moving averages and Bollinger Bands
2. **Indicators**: Shows technical indicators (RSI, MACD, Bollinger Bands, ADX)
3. **Settings**: Configure all trading parameters and strategy settings
4. **Positions**: View current open positions and their performance
5. **Logs**: Real-time log messages from the trading system

### Settings Panel
The Settings tab allows you to configure:
- Technical indicator parameters
- Risk management settings
- Order execution parameters
- Advanced strategy settings

All changes in the Settings panel can be saved to the configuration file for persistence.

## Troubleshooting

### Common Issues
- **API Connection Errors**: Verify your API keys and network connection
- **GUI Not Displaying**: Ensure tkinter and matplotlib are properly installed
- **No Trading Signals**: The bot requires sufficient historical data before generating signals

### Logs
- Check the `enhanced_bot_gui.log` file for detailed information
- The Logs tab in the GUI displays real-time log messages

## Advanced Usage

### Customizing Strategies
The bot uses the Master Omni Overlord Strategy by default, which combines:
- Triple Confluence Strategy (funding edge + order book imbalance + technical triggers)
- Oracle Update Strategy (capitalizing on Hyperliquid's oracle update cycle)
- Adaptive market regime detection

To modify or create new strategies, see the files in the `strategies/` directory.

### Backtesting
Use the backtesting tools to evaluate strategy performance:
```bash
python real_data_backtesting.py
```

## Safety Features

The bot includes several safety mechanisms:
- Circuit breakers to pause trading after consecutive losses
- Dynamic position sizing based on market volatility
- Tiered stop-loss strategy with partial exits
- Synthetic data detection and risk adjustments

## Support

For issues or questions, please refer to the documentation or contact support.
