"""
Training Execution Script

This script runs the comprehensive training and backtesting process for the enhanced trading bot.
"""

import os
import sys
import asyncio
import logging
import json
from datetime import datetime

# Import our modules
from data_collector import DataCollector
from backtest_trainer import BacktestTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training_execution.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("TrainingExecution")

async def run_training_cycle(cycle_num=1, symbols=None, days=30, interval="1h", optimize=True):
    """
    Run a complete training cycle.
    
    Args:
        cycle_num: Training cycle number
        symbols: List of symbols to train on
        days: Number of days of historical data
        interval: Data interval
        optimize: Whether to optimize strategy parameters
    """
    logger.info(f"Starting training cycle {cycle_num}...")
    
    # Default symbols if not provided
    if symbols is None:
        symbols = ["BTC-USD-PERP", "ETH-USD-PERP", "SOL-USD-PERP"]
    
    # Create cycle directory
    cycle_dir = f"training_cycles/cycle_{cycle_num}"
    os.makedirs(cycle_dir, exist_ok=True)
    
    # Step 1: Collect data
    logger.info("Step 1: Collecting data...")
    data_collector = DataCollector()
    collection_results = data_collector.collect_all_data(symbols, days, interval)
    
    with open(f"{cycle_dir}/collection_results.json", "w") as f:
        json.dump(collection_results, f, indent=2)
    
    # Step 2: Run backtesting and training
    logger.info("Step 2: Running backtesting and training...")
    trainer = BacktestTrainer("config.json")
    training_results = await trainer.run_full_training(symbols, days, interval, optimize)
    
    # Step 3: Evaluate results
    logger.info("Step 3: Evaluating results...")
    evaluation = evaluate_training_results(training_results)
    
    with open(f"{cycle_dir}/evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)
    
    # Step 4: Save cycle results
    logger.info("Step 4: Saving cycle results...")
    cycle_results = {
        "cycle_num": cycle_num,
        "timestamp": datetime.now().isoformat(),
        "symbols": symbols,
        "days": days,
        "interval": interval,
        "optimize": optimize,
        "collection_results": collection_results,
        "training_results": training_results,
        "evaluation": evaluation
    }
    
    with open(f"{cycle_dir}/cycle_results.json", "w") as f:
        json.dump(cycle_results, f, indent=2, default=str)
    
    logger.info(f"Training cycle {cycle_num} completed.")
    
    return evaluation

def evaluate_training_results(training_results):
    """
    Evaluate training results to determine if they meet success criteria.
    
    Args:
        training_results: Results from training
        
    Returns:
        Evaluation dictionary
    """
    # Initialize evaluation
    evaluation = {
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "metrics": {},
        "issues": [],
        "recommendations": []
    }
    
    try:
        # Extract results for each symbol and strategy
        symbol_results = training_results.get("results", {})
        
        # Track metrics across all symbols
        all_returns = []
        all_win_rates = []
        all_sharpe_ratios = []
        
        # Evaluate each symbol
        for symbol, results in symbol_results.items():
            symbol_metrics = {}
            
            # Check combined strategy results
            if "combined" in results:
                combined = results["combined"]
                total_return = combined.get("total_return", 0)
                win_rate = combined.get("win_rate", 0)
                sharpe_ratio = combined.get("sharpe_ratio", 0)
                
                symbol_metrics["combined"] = {
                    "total_return": total_return,
                    "win_rate": win_rate,
                    "sharpe_ratio": sharpe_ratio
                }
                
                all_returns.append(total_return)
                all_win_rates.append(win_rate)
                all_sharpe_ratios.append(sharpe_ratio)
                
                # Check for issues
                if total_return < 0.05:  # Less than 5% return
                    evaluation["issues"].append(f"{symbol} combined strategy has low return: {total_return:.2%}")
                
                if win_rate < 0.5:  # Less than 50% win rate
                    evaluation["issues"].append(f"{symbol} combined strategy has low win rate: {win_rate:.2%}")
                
                if sharpe_ratio < 1.0:  # Low Sharpe ratio
                    evaluation["issues"].append(f"{symbol} combined strategy has low Sharpe ratio: {sharpe_ratio:.2f}")
            
            # Check individual strategies
            for strategy in ["triple_confluence", "oracle_update"]:
                if strategy in results:
                    strat_results = results[strategy]
                    total_return = strat_results.get("total_return", 0)
                    win_rate = strat_results.get("win_rate", 0)
                    sharpe_ratio = strat_results.get("sharpe_ratio", 0)
                    
                    symbol_metrics[strategy] = {
                        "total_return": total_return,
                        "win_rate": win_rate,
                        "sharpe_ratio": sharpe_ratio
                    }
            
            evaluation["metrics"][symbol] = symbol_metrics
        
        # Calculate overall metrics
        if all_returns:
            avg_return = sum(all_returns) / len(all_returns)
            avg_win_rate = sum(all_win_rates) / len(all_win_rates)
            avg_sharpe_ratio = sum(all_sharpe_ratios) / len(all_sharpe_ratios)
            
            evaluation["metrics"]["overall"] = {
                "avg_return": avg_return,
                "avg_win_rate": avg_win_rate,
                "avg_sharpe_ratio": avg_sharpe_ratio
            }
            
            # Determine success
            success_criteria = [
                avg_return >= 0.1,  # At least 10% average return
                avg_win_rate >= 0.55,  # At least 55% win rate
                avg_sharpe_ratio >= 1.5  # At least 1.5 Sharpe ratio
            ]
            
            evaluation["success"] = all(success_criteria)
            
            # Generate recommendations
            if not evaluation["success"]:
                if avg_return < 0.1:
                    evaluation["recommendations"].append("Optimize for higher returns by adjusting take-profit levels")
                
                if avg_win_rate < 0.55:
                    evaluation["recommendations"].append("Improve win rate by refining entry criteria and signal confidence thresholds")
                
                if avg_sharpe_ratio < 1.5:
                    evaluation["recommendations"].append("Reduce volatility by implementing tighter stop-loss and position sizing")
        
    except Exception as e:
        evaluation["success"] = False
        evaluation["issues"].append(f"Error evaluating results: {str(e)}")
    
    return evaluation

async def run_iterative_training(max_cycles=5):
    """
    Run iterative training cycles until success criteria are met.
    
    Args:
        max_cycles: Maximum number of training cycles
    """
    logger.info(f"Starting iterative training with maximum {max_cycles} cycles...")
    
    # Create training cycles directory
    os.makedirs("training_cycles", exist_ok=True)
    
    # Define symbols to train on
    symbols = ["BTC-USD-PERP", "ETH-USD-PERP", "SOL-USD-PERP"]
    
    # Run initial cycle
    cycle_num = 1
    evaluation = await run_training_cycle(cycle_num, symbols)
    
    # Continue cycles until success or max_cycles reached
    while not evaluation["success"] and cycle_num < max_cycles:
        logger.info(f"Cycle {cycle_num} did not meet success criteria. Starting next cycle...")
        
        # Increment cycle number
        cycle_num += 1
        
        # Adjust parameters based on recommendations
        days = 30 + (cycle_num * 5)  # Increase historical data
        
        # Run next cycle
        evaluation = await run_training_cycle(cycle_num, symbols, days=days)
    
    # Final report
    if evaluation["success"]:
        logger.info(f"Training successful after {cycle_num} cycles!")
    else:
        logger.warning(f"Reached maximum {max_cycles} cycles without meeting all success criteria.")
    
    # Generate final report
    final_report = {
        "timestamp": datetime.now().isoformat(),
        "total_cycles": cycle_num,
        "success": evaluation["success"],
        "final_evaluation": evaluation,
        "training_complete": True
    }
    
    with open("training_cycles/final_report.json", "w") as f:
        json.dump(final_report, f, indent=2, default=str)
    
    logger.info("Iterative training completed. Final report generated.")
    
    return final_report

async def main():
    """Main entry point."""
    logger.info("Starting comprehensive training and backtesting process...")
    
    # Run iterative training
    final_report = await run_iterative_training(max_cycles=5)
    
    # Print final status
    if final_report["success"]:
        logger.info("TRAINING SUCCESSFUL: All success criteria met!")
    else:
        logger.warning("TRAINING COMPLETED: Some success criteria not met, but training cycles completed.")
    
    logger.info("Comprehensive training and backtesting process completed.")

if __name__ == "__main__":
    asyncio.run(main())
