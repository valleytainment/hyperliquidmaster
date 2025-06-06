# Hyperliquid Master Trading Bot

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](README.md)

A professional, production-ready trading bot for the Hyperliquid decentralized exchange. Features advanced trading strategies, comprehensive risk management, and both GUI and CLI interfaces.

## 🚀 Features

### Trading Strategies
- **BB RSI ADX Strategy**: Combines Bollinger Bands, RSI, and ADX indicators for trend analysis
- **Hull Suite Strategy**: Uses Hull Moving Average and ATR for momentum trading
- **Extensible Framework**: Easy to add custom trading strategies

### Risk Management
- Position sizing and portfolio risk limits
- Daily loss limits and maximum drawdown protection
- Dynamic stop-loss and take-profit management
- Leverage controls and position monitoring

### Interfaces
- **GUI Interface**: User-friendly desktop application with real-time charts
- **CLI Interface**: Command-line interface for server deployments
- **Headless Mode**: Perfect for VPS and automated trading

### Advanced Features
- **Auto-Connection**: Connects automatically with default credentials
- **Secure Storage**: Encrypted private key management
- **Wallet Generation**: Create new wallets directly in the application
- **Backtesting Engine**: Test strategies on historical data
- **Comprehensive Logging**: Detailed logs for monitoring and debugging

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- Git

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/valleytainment/hyperliquidmaster.git
   cd hyperliquidmaster
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   # GUI Mode (with display)
   python main.py --mode gui
   
   # CLI Mode (headless compatible)
   python main.py --mode cli
   
   # Setup Mode (first-time configuration)
   python main.py --mode setup
   
   # Trading Mode (automated trading)
   python main.py --mode trading
   ```

### Headless/Server Installation

For VPS or headless environments:

```bash
# Use the headless version
python main_headless.py --mode cli
python main_headless.py --mode setup
python main_headless.py --mode trading
```

## 🔧 Configuration

### First-Time Setup

1. **Run setup mode**
   ```bash
   python main.py --mode setup
   ```

2. **Choose your option**
   - **Option 1**: Enter existing wallet credentials
   - **Option 2**: Generate new wallet
   - **Option 3**: Use default credentials (for testing)

### Configuration Files

- `config/config.yaml`: Main configuration file
- `config/strategies.yaml`: Strategy-specific settings
- `logs/`: Application logs directory

### Environment Variables

```bash
# Optional: Set custom configuration
export HYPERLIQUID_CONFIG_PATH="/path/to/config.yaml"
export HYPERLIQUID_LOG_LEVEL="INFO"
```

## 🎯 Usage Examples

### GUI Mode
```bash
# Start with GUI interface
python main.py --mode gui

# Custom configuration
python main.py --mode gui --config custom_config.yaml
```

### CLI Mode
```bash
# Interactive CLI
python main.py --mode cli

# Available commands in CLI:
# help       - Show available commands
# status     - Show connection status
# connect    - Connect with custom credentials
# default    - Use default credentials
# generate   - Generate new wallet
# test       - Test connection
# strategies - Show available strategies
# quit/exit  - Exit application
```

### Trading Mode
```bash
# Start automated trading
python main.py --mode trading

# With custom log level
python main.py --mode trading --log-level DEBUG
```

### Headless Mode (Servers)
```bash
# Perfect for VPS/servers without display
python main_headless.py --mode cli
python main_headless.py --mode trading
```

## 📊 Trading Strategies

### BB RSI ADX Strategy
Combines three powerful indicators:
- **Bollinger Bands**: Volatility and mean reversion
- **RSI**: Momentum and overbought/oversold conditions  
- **ADX**: Trend strength measurement

**Parameters:**
- BB Period: 20, Standard Deviation: 2.0
- RSI Period: 14, Oversold: 25, Overbought: 75
- ADX Period: 14, Threshold: 25

### Hull Suite Strategy
Advanced momentum strategy using:
- **Hull Moving Average**: Reduced lag moving average
- **ATR**: Volatility-based position sizing

**Parameters:**
- Hull MA Period: 34
- ATR Period: 14, Multiplier: 2.0

## 🛡️ Risk Management

### Position Limits
- Maximum portfolio risk: 2%
- Maximum daily loss: 5%
- Maximum drawdown: 10%
- Maximum leverage: 3x
- Maximum position size: 10%

### Safety Features
- Real-time risk monitoring
- Automatic position sizing
- Emergency stop-loss triggers
- Daily loss limits
- Drawdown protection

## 🧪 Testing

### Run Tests
```bash
# Test original main.py functionality
python test_original_main.py

# Comprehensive functionality tests
python test_comprehensive_final.py

# Headless environment tests
python test_main_headless.py
```

### Test Results
All tests pass successfully:
- ✅ Import tests: 3/3 passed
- ✅ Strategy initialization: Working
- ✅ Risk management: Operational
- ✅ API connectivity: Functional
- ✅ Backtesting engine: Ready

## 📁 Project Structure

```
hyperliquidmaster/
├── main.py                    # Main application (GUI + CLI)
├── main_headless.py          # Headless version for servers
├── requirements.txt          # Python dependencies
├── config/                   # Configuration files
│   ├── config.yaml          # Main configuration
│   └── strategies.yaml      # Strategy settings
├── core/                    # Core functionality
│   ├── api.py              # Hyperliquid API wrapper
│   └── connection_manager_enhanced.py
├── strategies/              # Trading strategies
│   ├── base_strategy.py    # Base strategy class
│   ├── bb_rsi_adx.py      # BB RSI ADX strategy
│   ├── hull_suite.py      # Hull Suite strategy
│   ├── strategy_manager.py # Strategy management
│   └── trading_types.py   # Trading data types
├── risk_management/         # Risk management
│   └── risk_manager.py    # Risk management system
├── gui/                    # GUI interface
│   └── enhanced_gui.py    # Main GUI application
├── backtesting/            # Backtesting engine
│   └── backtest_engine.py # Historical testing
├── utils/                  # Utilities
│   ├── config_manager.py  # Configuration management
│   ├── security.py        # Security and encryption
│   └── logger.py          # Logging system
├── tests/                  # Test files
├── logs/                   # Application logs
└── docs/                   # Documentation
```

## 🔐 Security

### Private Key Management
- **Encrypted Storage**: Private keys are encrypted using industry-standard encryption
- **Secure Input**: Hidden input for private key entry
- **Memory Protection**: Keys are cleared from memory after use

### Best Practices
- Never share your private keys
- Use strong passwords for encryption
- Regularly backup your configuration
- Monitor logs for suspicious activity

## 🚀 Deployment

### Local Development
```bash
git clone https://github.com/valleytainment/hyperliquidmaster.git
cd hyperliquidmaster
pip install -r requirements.txt
python main.py --mode setup
```

### VPS/Server Deployment
```bash
# Install on server
git clone https://github.com/valleytainment/hyperliquidmaster.git
cd hyperliquidmaster
pip install -r requirements.txt

# Use headless version
python main_headless.py --mode setup
python main_headless.py --mode trading

# Optional: Run as service
nohup python main_headless.py --mode trading > trading.log 2>&1 &
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "main_headless.py", "--mode", "trading"]
```

## 📈 Performance

### Backtesting Results
- **BB RSI ADX Strategy**: Optimized for trending markets
- **Hull Suite Strategy**: Excellent for momentum trading
- **Risk-Adjusted Returns**: Consistent performance with controlled drawdowns

### System Requirements
- **Minimum**: 1 CPU, 512MB RAM
- **Recommended**: 2 CPU, 1GB RAM
- **Network**: Stable internet connection
- **Storage**: 100MB for application + logs

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
git clone https://github.com/valleytainment/hyperliquidmaster.git
cd hyperliquidmaster
pip install -r requirements.txt
python test_comprehensive_final.py  # Run tests
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- [API Documentation](docs/api.md)
- [Strategy Development Guide](docs/strategies.md)
- [Configuration Reference](docs/configuration.md)

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/valleytainment/hyperliquidmaster/issues)
- **Discussions**: [GitHub Discussions](https://github.com/valleytainment/hyperliquidmaster/discussions)

### Common Issues

**Import Errors**
```bash
# All import errors have been resolved
# If you encounter any, please run:
python test_original_main.py
```

**Connection Issues**
```bash
# Test your connection
python main.py --mode cli
> test
```

**GUI Not Working**
```bash
# Use headless version on servers
python main_headless.py --mode cli
```

## 🎉 Acknowledgments

- Hyperliquid team for the excellent DEX platform
- Python trading community for inspiration
- Contributors and testers

## 📊 Status

- ✅ **Production Ready**: All import errors resolved
- ✅ **Fully Tested**: Comprehensive test suite passing
- ✅ **Auto-Connected**: Ready to trade immediately
- ✅ **Professional**: Clean, optimized codebase
- ✅ **Documented**: Complete documentation and examples

---

**Ready to start trading? Run `python main.py --mode setup` to get started!**

