"""
Enhanced VWMA Indicator for Limited Data

This module provides an enhanced VWMA indicator that can work with limited data
and generate actionable signals even when full historical data is not available.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class EnhancedVWMAIndicator:
    """
    Enhanced VWMA indicator that can work with limited data.
    """
    
    @staticmethod
    def calculate_vwma(df: pd.DataFrame, periods: List[int] = [20, 50, 100], price_col: str = 'close', volume_col: str = 'volume') -> pd.DataFrame:
        """
        Calculate VWMA for multiple periods with enhanced handling for limited data.
        
        Args:
            df: DataFrame with price and volume data
            periods: List of VWMA periods to calculate
            price_col: Column name for price data
            volume_col: Column name for volume data
            
        Returns:
            DataFrame with VWMA columns added
        """
        try:
            # Make a copy to avoid modifying the original
            result_df = df.copy()
            
            # Check if volume column exists
            if volume_col not in result_df.columns:
                # Create synthetic volume based on price volatility
                logger.warning(f"Volume column '{volume_col}' not found, creating synthetic volume")
                
                # Calculate returns
                returns = result_df[price_col].pct_change().fillna(0)
                
                # Create synthetic volume (higher for higher absolute returns)
                result_df[volume_col] = 1000 * (1 + 10 * returns.abs())
                
            # Calculate VWMA for each period
            for period in periods:
                # Adjust period if not enough data
                adjusted_period = min(period, len(result_df))
                
                if adjusted_period < 2:
                    logger.warning(f"Not enough data for VWMA({period})")
                    result_df[f'vwma_{period}'] = result_df[price_col]
                    continue
                    
                # Calculate VWMA
                vwma = result_df[price_col].multiply(result_df[volume_col]).rolling(window=adjusted_period, min_periods=1).sum() / \
                       result_df[volume_col].rolling(window=adjusted_period, min_periods=1).sum()
                       
                result_df[f'vwma_{period}'] = vwma
                
            # Add VWMA crossover signals
            if 'vwma_20' in result_df.columns and 'vwma_50' in result_df.columns:
                result_df['vwma_cross'] = np.where(result_df['vwma_20'] > result_df['vwma_50'], 1, 
                                          np.where(result_df['vwma_20'] < result_df['vwma_50'], -1, 0))
                                          
            # Add VWMA trend signals
            for period in periods:
                col_name = f'vwma_{period}'
                if col_name in result_df.columns:
                    # Calculate trend (current VWMA vs. VWMA 5 periods ago)
                    result_df[f'{col_name}_trend'] = np.where(result_df[col_name] > result_df[col_name].shift(5), 1,
                                                     np.where(result_df[col_name] < result_df[col_name].shift(5), -1, 0))
                    
            return result_df
        except Exception as e:
            logger.error(f"Error calculating VWMA: {str(e)}")
            return df
            
    @staticmethod
    def check_vwma_crossover(df: pd.DataFrame, fast_period: int = 20, slow_period: int = 50) -> Tuple[bool, bool, float]:
        """
        Check for VWMA crossover with enhanced handling for limited data.
        
        Args:
            df: DataFrame with VWMA columns
            fast_period: Fast VWMA period
            slow_period: Slow VWMA period
            
        Returns:
            Tuple of (bullish_crossover, bearish_crossover, strength)
        """
        try:
            # Check if VWMA columns exist
            fast_col = f'vwma_{fast_period}'
            slow_col = f'vwma_{slow_period}'
            
            if fast_col not in df.columns or slow_col not in df.columns:
                logger.warning(f"Missing VWMA columns for crossover check: {fast_col} or {slow_col}")
                return False, False, 0.0
                
            # Get current and previous values
            if len(df) < 2:
                logger.warning("Not enough data points for VWMA crossover check")
                return False, False, 0.0
                
            current_fast = df[fast_col].iloc[-1]
            current_slow = df[slow_col].iloc[-1]
            
            # If we don't have enough history for previous values, use trend columns
            if 'vwma_cross' in df.columns:
                current_cross = df['vwma_cross'].iloc[-1]
                prev_cross = df['vwma_cross'].iloc[-2] if len(df) > 2 else 0
                
                bullish_crossover = prev_cross <= 0 and current_cross > 0
                bearish_crossover = prev_cross >= 0 and current_cross < 0
            else:
                # Fallback to current position
                bullish_crossover = current_fast > current_slow
                bearish_crossover = current_fast < current_slow
                
            # Calculate strength based on distance between VWMAs
            if current_slow != 0:
                strength = abs(current_fast - current_slow) / current_slow
            else:
                strength = 0.0
                
            # Enhance strength based on trend consistency
            if f'{fast_col}_trend' in df.columns and f'{slow_col}_trend' in df.columns:
                fast_trend = df[f'{fast_col}_trend'].iloc[-1]
                slow_trend = df[f'{slow_col}_trend'].iloc[-1]
                
                # If trends align with crossover, increase strength
                if (bullish_crossover and fast_trend > 0 and slow_trend > 0) or \
                   (bearish_crossover and fast_trend < 0 and slow_trend < 0):
                    strength *= 1.5
                    
            return bullish_crossover, bearish_crossover, strength
        except Exception as e:
            logger.error(f"Error checking VWMA crossover: {str(e)}")
            return False, False, 0.0
            
    @staticmethod
    def generate_vwma_signal(df: pd.DataFrame) -> Dict:
        """
        Generate trading signal based on VWMA with enhanced handling for limited data.
        
        Args:
            df: DataFrame with price and VWMA data
            
        Returns:
            Signal dictionary
        """
        try:
            # Calculate VWMA if not already present
            if 'vwma_20' not in df.columns or 'vwma_50' not in df.columns:
                df = EnhancedVWMAIndicator.calculate_vwma(df)
                
            # Check for VWMA crossover
            bullish, bearish, strength = EnhancedVWMAIndicator.check_vwma_crossover(df)
            
            # Generate signal
            if bullish:
                action = "long"
                signal = strength
            elif bearish:
                action = "short"
                signal = -strength
            else:
                # Check current position relative to VWMA
                if 'close' in df.columns and 'vwma_20' in df.columns:
                    current_price = df['close'].iloc[-1]
                    current_vwma = df['vwma_20'].iloc[-1]
                    
                    # Price significantly above VWMA
                    if current_price > current_vwma * 1.02:
                        action = "long"
                        signal = 0.3
                    # Price significantly below VWMA
                    elif current_price < current_vwma * 0.98:
                        action = "short"
                        signal = -0.3
                    else:
                        action = "none"
                        signal = 0.0
                else:
                    action = "none"
                    signal = 0.0
                    
            return {
                "action": action,
                "signal": signal,
                "strength": abs(signal),
                "indicator": "vwma",
                "bullish_crossover": bullish,
                "bearish_crossover": bearish
            }
        except Exception as e:
            logger.error(f"Error generating VWMA signal: {str(e)}")
            return {"action": "none", "signal": 0.0, "strength": 0.0}
