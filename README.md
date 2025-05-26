# Hyperliquid Trading Bot

A robust trading bot for the Hyperliquid exchange with advanced features including API rate limiting with persistent cooldown, mock data mode, and XRP-specific trading strategies.

## Features

- **API Rate Limiting with Long Cooldown**
  - Persistent cooldown state that survives across bot restarts
  - Automatic mock data mode activation during API rate limit periods

- **Mock Data Integration**
  - Realistic synthetic market data generation
  - Seamless operation during API rate limits

- **Advanced Trading Strategies**
  - Multi-timeframe signal generation
  - Adaptive parameters based on market conditions
  - Comprehensive technical indicators

- **Error Handling and Recovery**
  - Context-aware error handling
  - Automatic recovery mechanisms
  - Detailed error logging

- **User-Friendly GUI**
  - Real-time signal monitoring
  - Configurable strategy parameters
  - Automatic switching between live and mock data

## Repository Structure

```
hyperliquid_complete/
├── core/                 # Core functionality
│   ├── api_rate_limiter.py   # API rate limiting with cooldown
│   ├── error_handling.py     # Error handling and recovery
│   └── hyperliquid_adapter.py # Hyperliquid API adapter
├── data/                 # Data handling
│   └── mock_data_provider.py # Mock data generation
├── strategies/           # Trading strategies
│   ├── signal_generator.py   # Signal generation
│   └── master_strategy.py    # Master trading strategy
├── utils/                # Utility functions
│   └── technical_indicators.py # Technical indicators
├── gui/                  # Graphical user interface
│   └── main.py               # Main GUI application
├── tests/                # Tests
│   └── headless_test.py      # Headless GUI testing
└── docs/                 # Documentation
    ├── README.md             # General documentation
    └── XRP_STRATEGY.md       # XRP strategy documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/valleytainment/hyperliquidmaster.git
cd hyperliquidmaster
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### GUI Mode

To run the bot with the graphical user interface:

```bash
python gui/main.py
```

### Headless Mode

To run the bot in headless mode (without GUI):

```bash
python tests/headless_test.py
```

### Backtesting

To run backtesting on XRP data:

```bash
python strategies/backtest.py
```

## Configuration

The bot can be configured through the GUI or by editing the configuration files:

- Strategy parameters in `strategies/master_strategy.py`
- API rate limiting parameters in `core/api_rate_limiter.py`
- Mock data parameters in `data/mock_data_provider.py`

## API Rate Limiting

The bot implements a sophisticated API rate limiting system that:

1. Tracks API call rates for different endpoints
2. Enforces cooldown periods when rate limits are exceeded
3. Persists cooldown state across bot restarts
4. Automatically activates mock data mode during cooldowns

## Mock Data Mode

When API rate limits are exceeded, the bot automatically switches to mock data mode, which:

1. Generates realistic synthetic market data
2. Maintains trading strategy operation
3. Provides seamless transition between real and mock data
4. Allows for development and testing without API access

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
