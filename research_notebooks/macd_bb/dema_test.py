# This is necessary to recognize the modules
import os
import sys
from decimal import Decimal
import warnings

warnings.filterwarnings("ignore")

root_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
sys.path.append(root_path)

from core.backtesting import BacktestingEngine

backtesting = BacktestingEngine(root_path=root_path, load_cached_data=True)

import datetime
from decimal import Decimal

from controllers.directional_trading.dema_st_adx_controller import DemaSTADXControllerConfig

connector_name = "binance_perpetual"
trading_pair = "BTC-USDT"
candles_connector = "binance_perpetual"
candles_trading_pair = "BTC-USDT"
interval = "5m"
# Risk Management
total_amount_quote = 100
leverage = 5
executor_timeout = 60
enable_startup_entry = False

# Strategy parameters

#DEMA
dema_length = 200

#SuperTrend
supertrend_length = 12
supertrend_multiplier = 3.0

#ADX
adx_length = 14
adx_threshold_choppy = 20.0
adx_threshold_trending = 25.0
adx_threshold_extreme = 45.0
adx_slope_period = 5

#Position sizing based on ADX
enable_adx_position_sizing = True
position_size_weak_trend = 0.5
position_size_strong_trend = 1.0
position_size_extreme_trend = 0.7


# Backtest period - Using last 10 days of data
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=10)
start = int(start_date.timestamp())
end = int(end_date.timestamp())

print(f"📅 Backtesting period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Creating the instance of the configuration and the controller
config = DemaSTADXControllerConfig(
    interval=interval,
    connector_name=connector_name,
    trading_pair=trading_pair,
    candles_connector=candles_connector,
    candles_trading_pair=candles_trading_pair,
    leverage=leverage,
    dema_length=dema_length,
    supertrend_length=supertrend_length,
    supertrend_multiplier=supertrend_multiplier,
    
    # ADX Configuration
    adx_length=adx_length,
    adx_threshold_choppy=adx_threshold_choppy,
    adx_threshold_trending=adx_threshold_trending,
    adx_threshold_extreme=adx_threshold_extreme,
    adx_slope_period=adx_slope_period,
    
    # Position sizing based on ADX
    enable_adx_position_sizing=enable_adx_position_sizing,
    position_size_weak_trend=Decimal(str(position_size_weak_trend)),
    position_size_strong_trend=Decimal(str(position_size_strong_trend)),
    position_size_extreme_trend=Decimal(str(position_size_extreme_trend)),
    total_amount_quote=Decimal(str(total_amount_quote)),
    
    
    executor_timeout=executor_timeout,
    enable_startup_entry=enable_startup_entry,
)

# Running the backtesting this will output a backtesting result object that has built in methods to visualize the results

import asyncio

async def run_backtest():
    backtesting_result = await backtesting.run_backtesting(config, start, end, interval)
    
    # Let's see what is inside the backtesting results
    print(backtesting_result.get_results_summary())
    backtesting_result.get_backtesting_figure()
    
    return backtesting_result

# Run the async function
if __name__ == "__main__":
    backtesting_result = asyncio.run(run_backtest())