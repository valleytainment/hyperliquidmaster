# Hyperliquid Trading Bot - Integrated Documentation

## Overview
This document provides a comprehensive overview of the Hyperliquid Trading Bot, which has been integrated from multiple branches to ensure all features, settings, and GUI components are preserved.

## Repository Structure
The integrated repository follows a professional, modular structure:

- **config/**: Trading modes, risk metrics, and configuration settings
- **core/**: Core functionality including API rate limiting and error handling
- **data/**: Data handling and mock data providers
- **strategies/**: Trading strategies including XRP-specific implementations
- **gui/**: User interface components with optimized implementations
- **tests/**: Testing frameworks including headless GUI testing
- **utils/**: Utility functions and helper modules
- **docs/**: Documentation and strategy explanations

## Key Features

### 1. API Rate Limiting with Long Cooldown
- Enhanced rate limiter that persists cooldown state across bot restarts
- Automatic mock data mode activation during API rate limit periods

### 2. Mock Data Integration
- Enhanced mock data provider for seamless operation during API limits
- Fixed JSON serialization issues in synthetic trade generation

### 3. Multi-Mode Trading
- Paper Trading: Simulated trades with no real money at risk
- Live Trading: Real money trading with standard risk parameters
- Monitor Only: Signal generation without trade execution
- Aggressive: Higher risk/reward parameters
- Conservative: Lower risk/reward parameters

### 4. XRP-Specific Trading Strategies
- Optimized technical indicators for XRP's volatility profile
- Multi-timeframe confirmation system for higher probability trades
- Adaptive parameter adjustment based on market conditions

### 5. Advanced GUI
- Real-time monitoring and control interface
- Automatic mock data mode during API rate limits
- Comprehensive risk metrics and performance tracking

### 6. Robust Error Handling
- Context-aware error recovery mechanisms
- Graceful degradation during API outages
- Detailed logging and diagnostics

## Usage Instructions

### Running the GUI
```
python gui/gui_main_optimized_fixed.py
```

### Running Headless Tests
```
python tests/headless_gui_test.py
```

### Configuration
All trading modes and risk parameters can be adjusted in the config files:
- `config/mode_settings.json`: Trading mode parameters
- `config/risk_metrics.json`: Risk tracking metrics
- `config/trading_mode.json`: Current active trading mode

## Integration Notes
This integrated version combines features from multiple branches:
- Master branch: Core functionality and basic trading logic
- Robust-trading-main: Enhanced XRP trading strategies
- MVP-minimal: Essential configuration and optimized GUI
- Reorganized-structure: Professional directory organization

All settings, features, and GUI components have been preserved to ensure a complete and fully-functional trading system.
