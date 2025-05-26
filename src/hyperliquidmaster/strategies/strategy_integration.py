"""
Strategy Integration with Adaptive Indicators

This module integrates the adaptive indicators with the trading strategies
to ensure robust signal generation even with limited historical data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from strategies.adaptive_indicator import AdaptiveIndicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class StrategyIntegration:
    """
    Integrates adaptive indicators with trading strategies.
    """
    
    @staticmethod
    def apply_indicators(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """
        Apply all necessary technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with price and volume data
            config: Configuration dictionary
            
        Returns:
            DataFrame with added indicator columns
        """
        try:
            # Make a copy to avoid modifying the original
            result_df = df.copy()
            
            # Apply VWMA
            vwma_periods = [20, 50, 200]
            for period in vwma_periods:
                result_df[f'vwma_{period}'] = AdaptiveIndicator.vwma(
                    df=result_df, 
                    period=period
                )
                
            # Apply RSI
            result_df['rsi_14'] = AdaptiveIndicator.rsi(
                df=result_df, 
                period=14
            )
            
            # Apply Bollinger Bands
            upper, middle, lower = AdaptiveIndicator.bollinger_bands(
                df=result_df, 
                period=20
            )
            result_df['bb_upper'] = upper
            result_df['bb_middle'] = middle
            result_df['bb_lower'] = lower
            
            # Apply MACD
            macd_line, signal_line, histogram = AdaptiveIndicator.macd(
                df=result_df
            )
            result_df['macd_line'] = macd_line
            result_df['macd_signal'] = signal_line
            result_df['macd_histogram'] = histogram
            
            # Detect market regime
            result_df['market_regime'] = AdaptiveIndicator.detect_market_regime(
                df=result_df
            )
            
            # Detect support and resistance
            support, resistance = AdaptiveIndicator.detect_support_resistance(
                df=result_df
            )
            result_df['support_level'] = support
            result_df['resistance_level'] = resistance
            
            return result_df
        except Exception as e:
            logger.error(f"Error applying indicators: {str(e)}")
            return df
            
    @staticmethod
    def prepare_market_data(market_data: Dict) -> Dict:
        """
        Prepare market data for strategy use by adding technical indicators.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Enhanced market data dictionary
        """
        try:
            # Check if historical data is available
            if 'historical_data' not in market_data or market_data['historical_data'] is None:
                logger.warning("No historical data available for indicator calculation")
                return market_data
                
            # Get historical data
            df = market_data['historical_data']
            
            # Apply indicators
            enhanced_df = StrategyIntegration.apply_indicators(df, {})
            
            # Update market data
            market_data['historical_data'] = enhanced_df
            
            # Add latest indicator values to market data for easy access
            if not enhanced_df.empty:
                latest = enhanced_df.iloc[-1]
                
                market_data['indicators'] = {
                    'vwma_20': latest.get('vwma_20', None),
                    'vwma_50': latest.get('vwma_50', None),
                    'vwma_200': latest.get('vwma_200', None),
                    'rsi_14': latest.get('rsi_14', None),
                    'bb_upper': latest.get('bb_upper', None),
                    'bb_middle': latest.get('bb_middle', None),
                    'bb_lower': latest.get('bb_lower', None),
                    'macd_line': latest.get('macd_line', None),
                    'macd_signal': latest.get('macd_signal', None),
                    'macd_histogram': latest.get('macd_histogram', None),
                    'market_regime': latest.get('market_regime', 'unknown'),
                    'support_level': latest.get('support_level', None),
                    'resistance_level': latest.get('resistance_level', None)
                }
                
                # Add trend detection
                price = market_data.get('last_price', 0)
                vwma_20 = market_data['indicators'].get('vwma_20', price)
                vwma_50 = market_data['indicators'].get('vwma_50', price)
                
                if vwma_20 is not None and vwma_50 is not None:
                    if vwma_20 > vwma_50:
                        market_data['indicators']['trend'] = 'bullish'
                    elif vwma_20 < vwma_50:
                        market_data['indicators']['trend'] = 'bearish'
                    else:
                        market_data['indicators']['trend'] = 'neutral'
                else:
                    market_data['indicators']['trend'] = 'unknown'
                    
                # Add volatility classification
                if 'rsi_14' in market_data['indicators'] and market_data['indicators']['rsi_14'] is not None:
                    rsi = market_data['indicators']['rsi_14']
                    if rsi > 70:
                        market_data['indicators']['overbought'] = True
                    elif rsi < 30:
                        market_data['indicators']['oversold'] = True
                    else:
                        market_data['indicators']['overbought'] = False
                        market_data['indicators']['oversold'] = False
                        
            return market_data
        except Exception as e:
            logger.error(f"Error preparing market data: {str(e)}")
            return market_data
