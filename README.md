# 🏆 Hyperliquid Trading Bot - BEST BRANCH EDITION

🚀 **The ULTIMATE and most comprehensive Hyperliquid trading bot** - This is the BEST BRANCH with maximum optimizations, enhanced features, and production-ready performance.

## 🌟 Features

### 🎯 **Core Trading Capabilities**
- **Multi-Strategy Support**: BB RSI ADX, Hull Suite, and extensible framework for custom strategies
- **Real-time Trading**: Live market data processing with sub-second execution
- **Advanced Order Management**: Market, limit, stop-loss, and take-profit orders
- **Portfolio Management**: Multi-asset portfolio tracking and optimization

### 🛡️ **Risk Management**
- **Comprehensive Risk Controls**: Position sizing, leverage limits, drawdown protection
- **Real-time Monitoring**: VaR calculation, correlation analysis, concentration risk
- **Emergency Stop**: Automatic trading halt on critical risk levels
- **Daily/Hourly Limits**: Trade frequency and loss limits

### 📊 **Backtesting & Analytics**
- **Historical Testing**: Comprehensive backtesting with realistic market simulation
- **Performance Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown, profit factor
- **Trade Analysis**: Detailed trade logs with entry/exit reasons and confidence scores
- **Equity Curve**: Visual portfolio performance tracking

### 🖥️ **Modern GUI Interface**
- **Real-time Dashboard**: Live portfolio monitoring and trading controls
- **Strategy Management**: Easy strategy configuration and monitoring
- **Risk Dashboard**: Visual risk metrics and alerts
- **Backtesting Interface**: Interactive backtesting with results visualization

### 🔒 **Security & Configuration**
- **Secure Key Management**: Encrypted private key storage
- **Testnet Support**: Safe testing environment
- **Flexible Configuration**: YAML-based configuration with validation
- **Comprehensive Logging**: Detailed logs for trading, errors, and performance

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Hyperliquid account (testnet or mainnet)
- Private key for your wallet

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd hyperliquid_integrated
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Initial setup**
```bash
python main.py --mode setup
```
This will guide you through:
- Private key setup (encrypted storage)
- Wallet address configuration
- Network selection (testnet/mainnet)

4. **Start the GUI**
```bash
python main.py --mode gui
```

### Alternative Run Modes

**Automated Trading Mode**
```bash
python main.py --mode trading
```

**Backtesting Mode**
```bash
python main.py --mode backtest
```

**Custom Configuration**
```bash
python main.py --config custom_config.yaml --log-level DEBUG
```

## 📋 Configuration

### Main Configuration (`config/trading_config.yaml`)

```yaml
trading:
  testnet: true
  wallet_address: "your_wallet_address"
  max_leverage: 10.0
  max_daily_loss: 5.0  # 5%
  max_drawdown: 10.0   # 10%
  default_order_size: 100  # USD
  active_strategies:
    - bb_rsi_adx
    - hull_suite

strategies:
  bb_rsi_adx:
    enabled: true
    indicators:
      bollinger_bands:
        period: 20
        std_dev: 2.0
      rsi:
        period: 14
        overbought: 75
        oversold: 25
      adx:
        period: 14
        threshold: 25
    
  hull_suite:
    enabled: true
    indicators:
      hull_ma:
        period: 34
        source: "close"
      atr:
        period: 14
        multiplier: 2.0
```

### Risk Management Configuration

```yaml
risk_management:
  max_portfolio_risk: 0.02      # 2% max risk per trade
  max_daily_loss: 0.05          # 5% max daily loss
  max_drawdown: 0.10            # 10% max drawdown
  max_leverage: 10.0            # 10x max leverage
  max_position_size: 0.20       # 20% max position size
  max_correlation: 0.7          # Max correlation between positions
  var_limit: 0.03               # 3% VaR limit
```

## 🎯 Trading Strategies

### 1. BB RSI ADX Strategy
**Bollinger Bands + RSI + ADX** trend-following strategy

**Entry Conditions:**
- **Long**: Price touches lower BB + RSI oversold + ADX > threshold
- **Short**: Price touches upper BB + RSI overbought + ADX > threshold

**Exit Conditions:**
- Price crosses middle BB (opposite direction)
- RSI reaches opposite extreme
- Stop loss or take profit hit

### 2. Hull Suite Strategy
**Hull Moving Average** trend identification with ATR-based stops

**Entry Conditions:**
- **Long**: Price crosses above Hull MA + Hull MA trending up (green)
- **Short**: Price crosses below Hull MA + Hull MA trending down (red)

**Exit Conditions:**
- Hull MA changes color (trend reversal)
- ATR-based stop loss hit
- Take profit target reached

## 📊 Performance Metrics

### Backtesting Results (Example)
```
Strategy: BB RSI ADX
Period: 2023-01-01 to 2024-01-01
Initial Capital: $10,000

Total Return: +24.5%
Max Drawdown: -8.2%
Sharpe Ratio: 1.85
Win Rate: 64.3%
Total Trades: 127
Profit Factor: 1.67
```

## 🛡️ Risk Management Features

### Position Sizing
- **Kelly Criterion**: Optimal position sizing based on win rate and average win/loss
- **Fixed Percentage**: Configurable percentage of portfolio per trade
- **Volatility Adjusted**: Position size adjusted based on asset volatility

### Risk Monitoring
- **Real-time VaR**: 1-day and 7-day Value at Risk calculation
- **Correlation Matrix**: Monitor correlation between positions
- **Concentration Risk**: Prevent over-exposure to single assets
- **Drawdown Tracking**: Real-time maximum drawdown monitoring

### Emergency Controls
- **Circuit Breakers**: Automatic trading halt on excessive losses
- **Daily Limits**: Maximum trades and losses per day
- **Leverage Limits**: Prevent excessive leverage usage
- **Volatility Filters**: Avoid trading in extreme market conditions

## 🔧 Advanced Features

### Custom Strategy Development
```python
from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType

class MyCustomStrategy(BaseStrategy):
    async def generate_signal(self, coin: str, market_data: List[MarketData]) -> TradingSignal:
        # Your custom logic here
        return TradingSignal(
            signal_type=SignalType.LONG,
            coin=coin,
            confidence=0.8,
            entry_price=current_price,
            stop_loss=stop_price,
            take_profit=target_price
        )
```

### API Integration
```python
from core.api import EnhancedHyperliquidAPI

api = EnhancedHyperliquidAPI(testnet=True)
api.authenticate(private_key, wallet_address)

# Get account state
account = api.get_account_state()

# Place order
order_result = api.place_order('BTC', 'buy', 100, order_type='market')

# Get market data
candles = api.get_candles('BTC', '15m', limit=100)
```

### Backtesting Framework
```python
from backtesting.backtest_engine import BacktestEngine
from strategies.bb_rsi_adx import BBRSIADXStrategy

# Initialize backtest
engine = BacktestEngine(initial_capital=10000)
strategy = BBRSIADXStrategy()

# Run backtest
results = await engine.run_backtest(
    strategy=strategy,
    market_data=historical_data,
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1)
)

# Analyze results
print(f"Total Return: {results.total_pnl_percentage:.2f}%")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown_percentage:.2f}%")
```

## 📁 Project Structure

```
hyperliquid_integrated/
├── main.py                     # Main application entry point
├── requirements.txt            # Python dependencies
├── config/                     # Configuration files
│   ├── trading_config.yaml     # Main trading configuration
│   └── logging_config.yaml     # Logging configuration
├── core/                       # Core API and utilities
│   ├── api.py                  # Enhanced Hyperliquid API wrapper
│   └── __init__.py
├── strategies/                 # Trading strategies
│   ├── base_strategy.py        # Base strategy framework
│   ├── bb_rsi_adx.py          # BB RSI ADX strategy
│   ├── hull_suite.py          # Hull Suite strategy
│   └── __init__.py
├── gui/                        # GUI interface
│   ├── enhanced_gui.py         # Main GUI application
│   └── __init__.py
├── backtesting/               # Backtesting framework
│   ├── backtest_engine.py     # Backtesting engine
│   └── __init__.py
├── risk_management/           # Risk management system
│   ├── risk_manager.py        # Risk manager
│   └── __init__.py
├── utils/                     # Utility modules
│   ├── logger.py              # Logging utilities
│   ├── config_manager.py      # Configuration management
│   ├── security.py            # Security utilities
│   └── __init__.py
├── data/                      # Data storage
├── logs/                      # Log files
├── tests/                     # Unit tests
└── docs/                      # Documentation
```

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/ -v
```

### Integration Tests
```bash
python -m pytest tests/integration/ -v
```

### Strategy Testing
```bash
python main.py --mode backtest
```

## 📝 Logging

### Log Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General information about bot operation
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failures

### Log Files
- `logs/trading.log`: Main trading log
- `logs/errors.log`: Error log
- `logs/performance.log`: Performance metrics log
- `logs/risk.log`: Risk management log

### Example Log Output
```
2024-01-15 10:30:15 [INFO] BB RSI ADX Signal: BTC LONG (confidence: 0.85, RSI: 23.4, ADX: 28.7, BB: lower)
2024-01-15 10:30:16 [INFO] Order placed: BTC buy $500 @ $42,150.00
2024-01-15 10:30:16 [INFO] Position opened: BTC long $500 (stop: $41,000, target: $44,000)
```

## ⚠️ Important Notes

### Security
- **Never share your private key** or store it in plain text
- **Use testnet first** to familiarize yourself with the bot
- **Start with small amounts** when moving to mainnet
- **Monitor the bot regularly** especially during volatile markets

### Risk Disclaimer
- **Trading cryptocurrencies involves significant risk** of loss
- **Past performance does not guarantee future results**
- **Only trade with money you can afford to lose**
- **The bot is provided as-is without warranty**

### Testnet vs Mainnet
- **Always test on testnet first** before using real funds
- **Testnet uses fake money** for safe testing
- **Switch to mainnet only after thorough testing**
- **Use small amounts initially** on mainnet

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd hyperliquid_integrated

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python -m pytest
```

### Adding New Strategies
1. Create new strategy file in `strategies/`
2. Inherit from `BaseStrategy`
3. Implement `generate_signal()` method
4. Add strategy configuration
5. Register in main application

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add comprehensive docstrings
- Write unit tests for new features

## 📞 Support

### Documentation
- **API Documentation**: See `docs/api.md`
- **Strategy Guide**: See `docs/strategies.md`
- **Configuration Guide**: See `docs/configuration.md`

### Community
- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Ask questions and share experiences

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project combines and enhances features from multiple open-source Hyperliquid trading bot repositories:

- **hyperliquid-python-sdk**: Official Hyperliquid Python SDK
- **Hyperliquid-Trading-Bot**: Discord-based trading bot by Sakaar-Sen
- **HyperLiquidAlgoBot**: High-frequency trading bot by SimSimButDifferent

Special thanks to the Hyperliquid team and the open-source community for their contributions.

---

**⚡ Ready to start trading? Run `python main.py --mode setup` to get started!**

