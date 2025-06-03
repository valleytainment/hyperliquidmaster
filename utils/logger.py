"""
Advanced logging configuration for the trading bot
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT
    }
    
    def format(self, record):
        # Add color to the log level
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        
        # Add color to the message based on content
        message = super().format(record)
        
        # Highlight important keywords
        if 'ERROR' in message or 'FAILED' in message.upper():
            message = f"{Fore.RED}{message}{Style.RESET_ALL}"
        elif 'SUCCESS' in message.upper() or 'COMPLETED' in message.upper():
            message = f"{Fore.GREEN}{message}{Style.RESET_ALL}"
        elif 'WARNING' in message or 'WARN' in message.upper():
            message = f"{Fore.YELLOW}{message}{Style.RESET_ALL}"
        
        return message


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration for the entire application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    console_format = ColoredFormatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        
        file_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(lineno)-4d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    
    # Daily rotating file handler
    from logging.handlers import TimedRotatingFileHandler
    
    daily_handler = TimedRotatingFileHandler(
        log_dir / f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log",
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    daily_handler.setLevel(logging.INFO)
    
    daily_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    daily_handler.setFormatter(daily_format)
    root_logger.addHandler(daily_handler)
    
    # Error file handler
    error_handler = logging.FileHandler(log_dir / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(daily_format)
    root_logger.addHandler(error_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('websocket').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TradingLogger:
    """Specialized logger for trading operations"""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.trade_file = Path("logs") / "trades.log"
        self.performance_file = Path("logs") / "performance.log"
        
        # Ensure logs directory exists
        self.trade_file.parent.mkdir(exist_ok=True)
    
    def log_trade(self, action: str, coin: str, side: str, size: float, 
                  price: float, pnl: float = None, **kwargs):
        """Log trading activity"""
        trade_data = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'coin': coin,
            'side': side,
            'size': size,
            'price': price,
            'pnl': pnl,
            **kwargs
        }
        
        # Log to main logger
        self.logger.info(f"TRADE: {action} {coin} {side} {size} @ {price}" + 
                        (f" PnL: {pnl}" if pnl else ""))
        
        # Log to trade file
        with open(self.trade_file, 'a', encoding='utf-8') as f:
            f.write(f"{trade_data}\n")
    
    def log_performance(self, metric: str, value: float, period: str = None):
        """Log performance metrics"""
        perf_data = {
            'timestamp': datetime.now().isoformat(),
            'metric': metric,
            'value': value,
            'period': period
        }
        
        self.logger.info(f"PERFORMANCE: {metric} = {value}" + 
                        (f" ({period})" if period else ""))
        
        with open(self.performance_file, 'a', encoding='utf-8') as f:
            f.write(f"{perf_data}\n")
    
    def log_signal(self, strategy: str, coin: str, signal: str, confidence: float = None):
        """Log trading signals"""
        signal_data = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'coin': coin,
            'signal': signal,
            'confidence': confidence
        }
        
        self.logger.info(f"SIGNAL: {strategy} -> {coin} {signal}" + 
                        (f" (confidence: {confidence})" if confidence else ""))
    
    def log_error(self, error: str, context: dict = None):
        """Log errors with context"""
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'error': error,
            'context': context or {}
        }
        
        self.logger.error(f"ERROR: {error}")
        if context:
            self.logger.error(f"Context: {context}")


# Initialize logging on module import
setup_logging()

