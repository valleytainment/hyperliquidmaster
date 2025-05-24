# Hyperliquid Trading Bot Testing Guide

This document provides instructions for testing the Hyperliquid trading bot in both GUI and headless modes.

## Prerequisites

- Python 3.8 or higher
- Required Python packages (install via `pip install -r requirements.txt`)
- Tkinter for GUI mode (usually comes with Python, but may need separate installation on some systems)

## Testing the Bot

### GUI Mode Testing

1. Run the GUI application:
   ```bash
   python gui_main.py
   ```

2. Verify the following components:
   - Main window loads with all buttons and controls
   - Price charts display correctly
   - Configuration controls work properly
   - Connection to Hyperliquid API succeeds
   - Real-time data updates are displayed

3. Test trading functionality:
   - Manual buy/sell buttons work
   - Position monitoring updates correctly
   - Stop-loss and take-profit settings are applied

### Headless Mode Testing

For environments without a display or for automated testing:

1. Run the headless test script:
   ```bash
   python headless_test.py
   ```

2. Review the test results in `headless_test_results.json`

3. Check the log file `headless_test.log` for any errors or warnings

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'X'**
   - Ensure all required packages are installed
   - Check that `__init__.py` files exist in all package directories

2. **TclError: no display name and no $DISPLAY environment variable**
   - This occurs when trying to run GUI mode in a headless environment
   - Use `headless_test.py` instead for testing in environments without displays

3. **API Rate Limiting Errors**
   - The bot includes rate limiting protection, but excessive testing may still trigger API limits
   - Wait a few minutes before retrying if you encounter rate limit errors

4. **JSON Serialization Errors**
   - If you encounter serialization errors, check that the data types being serialized are supported
   - The bot includes a custom JSON encoder to handle most common data types

## Reporting Issues

If you encounter any issues not covered in this guide, please:
1. Check the log files for error messages
2. Verify your configuration settings
3. Ensure you have the latest version of the code from the repository

## Running in Production

For production use:
1. Configure your API keys and trading parameters in `config.json`
2. Run the bot in either GUI or headless mode depending on your environment
3. Monitor the logs for any warnings or errors
4. Start with small position sizes until you've verified the bot's performance
