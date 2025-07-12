from typing import List
from decimal import Decimal

import pandas as pd
import pandas_ta as ta
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from hummingbot.core.data_type.common import TradeType, PriceType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction, CreateExecutorAction, StopExecutorAction


class DemaSuperTrendConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "dema_supertrend"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the connector for the candles data, leave empty to use the same exchange as the connector: ",
            "prompt_on_new": True})
    candles_trading_pair: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the trading pair for the candles data, leave empty to use the same trading pair as the connector: ",
            "prompt_on_new": True})
    interval: str = Field(
        default="3m",
        json_schema_extra={"prompt": "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ", "prompt_on_new": True})
    dema_length: int = Field(
        default=200,
        json_schema_extra={"prompt": "Enter the DEMA length: ", "prompt_on_new": True})
    supertrend_length: int = Field(
        default=12,
        json_schema_extra={"prompt": "Enter the SuperTrend length: ", "prompt_on_new": True})
    supertrend_multiplier: float = Field(
        default=3.0,
        json_schema_extra={"prompt": "Enter the SuperTrend multiplier: ", "prompt_on_new": True})
    candles_length: int = Field(
        default=15,
        json_schema_extra={"prompt": "Enter the minimum candles length: ", "prompt_on_new": True})

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v


class DemaSuperTrendController(DirectionalTradingControllerBase):
    def __init__(self, config: DemaSuperTrendConfig, *args, **kwargs):
        self.config = config
        self.max_records = max(config.dema_length, config.supertrend_length, config.candles_length) + 20
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(connector_name=self.config.candles_connector,
                                                      trading_pair=self.config.candles_trading_pair,
                                                      interval=self.config.interval,
                                                      max_records=self.max_records)
        if df is None or df.empty:
            print(f"[DEMA SuperTrend] WARNING: No candle data available")
            self.processed_data["signal"] = 0
            self.processed_data["features"] = df
            return
        
        print(f"[DEMA SuperTrend] Got {len(df)} candles, requesting max_records={self.max_records}")
        print(f"[DEMA SuperTrend] DataFrame columns before indicators: {df.columns.tolist()}")
        
        # Add indicators
        df.ta.dema(length=self.config.dema_length, append=True)
        df.ta.supertrend(length=self.config.supertrend_length, multiplier=self.config.supertrend_multiplier, append=True)
        
        print(f"[DEMA SuperTrend] DataFrame columns after indicators: {df.columns.tolist()}")
        
        # Check if indicators were calculated
        dema_col = f"DEMA_{self.config.dema_length}"
        # Handle float multiplier in column name (pandas_ta might format it differently)
        supertrend_col = f"SUPERTd_{self.config.supertrend_length}_{self.config.supertrend_multiplier}"
        
        # Check for alternative column name formats
        if supertrend_col not in df.columns:
            # Try with formatted float
            alt_supertrend_col = f"SUPERTd_{self.config.supertrend_length}_{self.config.supertrend_multiplier:.1f}"
            if alt_supertrend_col in df.columns:
                supertrend_col = alt_supertrend_col
                print(f"[DEMA SuperTrend] Using alternative SuperTrend column: {supertrend_col}")
        
        if dema_col not in df.columns:
            print(f"[DEMA SuperTrend] ERROR: DEMA indicator not calculated! Missing column: {dema_col}")
            self.processed_data["signal"] = 0
            self.processed_data["features"] = df
            return
            
        if supertrend_col not in df.columns:
            print(f"[DEMA SuperTrend] ERROR: SuperTrend indicator not calculated! Missing column: {supertrend_col}")
            self.processed_data["signal"] = 0
            self.processed_data["features"] = df
            return
        
        # Check for NaN values
        print(f"[DEMA SuperTrend] DEMA NaN count: {df[dema_col].isna().sum()}")
        print(f"[DEMA SuperTrend] SuperTrend NaN count: {df[supertrend_col].isna().sum()}")
        
        # Generate signals for ALL candles (vectorized approach for backtesting)
        # Use simple shift operation to avoid creating extra NaN columns
        prev_supertrend = df[supertrend_col].shift(1)
        
        # Long Entry: Price above DEMA AND SuperTrend turns green
        long_condition = (
            (df['close'] > df[dema_col]) & 
            (df[supertrend_col] == 1) & 
            (prev_supertrend == -1)
        )
        
        # Short Entry: Price below DEMA AND SuperTrend turns red  
        short_condition = (
            (df['close'] < df[dema_col]) & 
            (df[supertrend_col] == -1) & 
            (prev_supertrend == 1)
        )
        
        # Generate signals for all candles
        df['signal'] = 0
        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1
        
        # Instead of dropping all NaN rows, only drop rows where critical columns are NaN
        # Keep rows where only the first few DEMA values are NaN
        essential_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'signal']
        df_clean = df.dropna(subset=['timestamp', 'close', 'SUPERTd_12_3.0']).copy()
        print(f"[DEMA SuperTrend] Cleaned DataFrame: {len(df_clean)}/{len(df)} rows ({len(df_clean)/len(df)*100:.1f}% retained)")
        
        if len(df_clean) == 0:
            print(f"[DEMA SuperTrend] ERROR: No clean rows available after selective dropna()!")
            self.processed_data["signal"] = 0
            self.processed_data["features"] = pd.DataFrame()
            return
            
        # Use cleaned dataframe
        df = df_clean
        
        # Get current values for logging
        current_price = df["close"].iloc[-1]
        current_dema = df[dema_col].iloc[-1]
        current_supertrend_direction = df[supertrend_col].iloc[-1]
        current_signal = df['signal'].iloc[-1]
        
        # Count signals
        total_long_signals = (df['signal'] == 1).sum()
        total_short_signals = (df['signal'] == -1).sum()
        total_signals = total_long_signals + total_short_signals
        
        print(f"[DEMA SuperTrend] Current - Price: {current_price:.4f}, DEMA: {current_dema:.4f}, ST_dir: {current_supertrend_direction}, Signal: {current_signal}")
        print(f"[DEMA SuperTrend] Total signals generated: {total_signals} (Long: {total_long_signals}, Short: {total_short_signals})")
        
        # Log some sample signals for debugging
        if total_signals > 0:
            signal_rows = df[df['signal'] != 0].tail(5)
            print(f"[DEMA SuperTrend] Last 5 signals:")
            for idx, row in signal_rows.iterrows():
                signal_type = "LONG" if row['signal'] == 1 else "SHORT"
                ts = pd.to_datetime(row['timestamp'], unit='s')
                print(f"  {idx}: {signal_type} - Time: {ts} (TS: {row['timestamp']}), Price: {row['close']:.2f}, DEMA: {row[dema_col]:.2f}, ST: {row[supertrend_col]}")
                
        # Check data range and format
        first_ts = pd.to_datetime(df['timestamp'].iloc[0], unit='s')
        last_ts = pd.to_datetime(df['timestamp'].iloc[-1], unit='s')
        print(f"[DEMA SuperTrend] Data range: {first_ts} to {last_ts}")
        print(f"[DEMA SuperTrend] Timestamp range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        print(f"[DEMA SuperTrend] DataFrame shape: {df.shape}")
        print(f"[DEMA SuperTrend] Signal column exists: {'signal' in df.columns}")
        
        # Check if our signals are in the backtesting time range
        backtest_start = 1751274000  # 2025-07-01 00:00:00
        backtest_end = 1751965200    # 2025-07-10 00:00:00  
        signals_in_backtest_range = df[(df['timestamp'] >= backtest_start) & (df['timestamp'] <= backtest_end) & (df['signal'] != 0)]
        print(f"[DEMA SuperTrend] Signals within backtest time range: {len(signals_in_backtest_range)}")

        # CRITICAL: Ensure features DataFrame has NO NaN values for backtesting engine
        # The backtesting engine calls dropna() which will remove ALL rows if ANY column has NaN
        df_clean_for_backtesting = df.copy()
        
        # Double-check: remove ALL problematic columns that might have NaN
        problematic_cols_final = ['SUPERTl_12_3.0', 'SUPERTs_12_3.0', 'SUPERT_12_3.0']  
        df_clean_for_backtesting = df_clean_for_backtesting.drop(columns=problematic_cols_final, errors='ignore')
        
        # CRITICAL: Forward fill ALL remaining NaN values in the final dataframe
        print(f"  Pre-fill NaN analysis:")
        for col in df_clean_for_backtesting.columns:
            nan_count = df_clean_for_backtesting[col].isna().sum()
            if nan_count > 0:
                print(f"    {col}: {nan_count} NaN values")
        
        # Apply aggressive NaN filling
        df_clean_for_backtesting = df_clean_for_backtesting.fillna(method='ffill').fillna(method='bfill')
        # If any NaN still remain, fill with 0 (last resort)
        df_clean_for_backtesting = df_clean_for_backtesting.fillna(0)
        
        # Verify no NaN values remain
        nan_check = df_clean_for_backtesting.isna().any().any()
        print(f"  Features DataFrame final NaN check: {nan_check}")
        if nan_check:
            print("  ERROR: NaN values still present! Columns with NaN:")
            for col in df_clean_for_backtesting.columns:
                nan_count = df_clean_for_backtesting[col].isna().sum()
                if nan_count > 0:
                    print(f"    {col}: {nan_count} NaN values")
        else:
            print("  ✓ Features DataFrame is completely clean for backtesting engine")
        
        # Update processed data (following the same pattern as vanya.py)
        self.processed_data["signal"] = int(current_signal)
        self.processed_data["features"] = df_clean_for_backtesting  # Use the clean version
        self.processed_data["current_price"] = current_price
        self.processed_data["current_dema"] = current_dema
        self.processed_data["current_supertrend_direction"] = current_supertrend_direction
        
        # CRITICAL: Verify features DataFrame is properly formatted for backtesting engine
        print(f"  Final features DataFrame check:")
        print(f"    Index type: {type(df.index)}")
        print(f"    Index values (first 3): {df.index[:3].tolist()}")
        print(f"    Timestamp column type: {df['timestamp'].dtype}")
        print(f"    Required columns present: {all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'signal'])}")
        print(f"    DataFrame is not empty: {not df.empty}")
        print(f"    No NaN values: {not df.isna().any().any()}")
        
        # Debug features dataframe for backtesting engine
        print(f"[DEMA SuperTrend] Features DataFrame for backtesting:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Has signal column: {'signal' in df.columns}")
        print(f"  Signal value range: {df['signal'].min()} to {df['signal'].max()}")
        print(f"  Non-zero signals: {(df['signal'] != 0).sum()}")
        sample_with_signals = df[df['signal'] != 0].head(3)
        if not sample_with_signals.empty:
            print(f"  Sample signals:")
            for idx, row in sample_with_signals.iterrows():
                print(f"    TS: {row['timestamp']}, Signal: {row['signal']}")
        print(f"  Setting processed_data['signal'] = {int(current_signal)}")
        
        # Detailed NaN analysis
        print(f"  Detailed NaN analysis:")
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"    {col}: {nan_count}/{len(df)} NaN values")
                
        # Check if the issue is SUPERT indicators creating too many NaN
        print(f"  First 10 rows of SUPERT indicators:")
        supert_cols = [col for col in df.columns if 'SUPERT' in col]
        for i in range(min(10, len(df))):
            values = [f"{col}={df[col].iloc[i]}" for col in supert_cols]
            print(f"    Row {i}: {', '.join(values)}")
            
        # The real fix: remove problematic columns that create unnecessary NaN
        # We only need SUPERTd (direction), not the upper/lower bounds
        problematic_cols = ['SUPERTl_12_3.0', 'SUPERTs_12_3.0', 'SUPERT_12_3.0']
        df = df.drop(columns=problematic_cols, errors='ignore')
        print(f"  After removing problematic SUPERT columns: {df.shape}")
        
        # CRITICAL FIX: Forward fill remaining NaN values to ensure backtesting engine doesn't drop all rows
        print(f"  NaN analysis before forward fill:")
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"    {col}: {nan_count}/{len(df)} NaN values")
        
        # Forward fill ALL columns with NaN values  
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"  NaN analysis after forward/backward fill:")
        remaining_nan_cols = []
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"    {col}: {nan_count}/{len(df)} NaN values")
                remaining_nan_cols.append(col)
        
        # Reset index to ensure proper iteration in backtesting engine
        df = df.reset_index(drop=True)
        print(f"  Reset index - new index range: {df.index[0]} to {df.index[-1]}")
        
        # Now check clean rows  
        clean_rows = df.dropna()
        print(f"  Rows after dropna(): {len(clean_rows)}/{len(df)} ({len(clean_rows)/len(df)*100:.1f}%)")
        
        if len(clean_rows) == len(df):
            signals_in_clean = (clean_rows['signal'] != 0).sum()
            print(f"  Signals in clean rows: {signals_in_clean}")
            print(f"  SUCCESS: ALL data is now clean for backtesting engine!")
        else:
            print(f"  ERROR: Still have NaN values that will cause backtesting engine dropna() to fail")
            print(f"  Remaining NaN columns: {remaining_nan_cols}")

    def create_actions_proposal(self) -> List[ExecutorAction]:
        """
        Override with comprehensive debugging for signal processing chain.
        """
        signal = self.processed_data.get("signal", "KEY_MISSING")
        timestamp = self.market_data_provider.time()
        current_time = pd.to_datetime(timestamp, unit='s')
        
        # Debug signal value and type
        print(f"[DEMA SuperTrend] create_actions_proposal - Signal: {signal} (type: {type(signal)}), Time: {current_time}")
        print(f"  processed_data keys: {list(self.processed_data.keys())}")
        
        if signal != 0 and signal != "KEY_MISSING":
            print(f"  Signal condition passed: {signal} != 0")
            can_create = self.can_create_executor(signal)
            print(f"  can_create_executor returned: {can_create}")
            
            if can_create:
                price = self.market_data_provider.get_price_by_type(
                    self.config.connector_name, self.config.trading_pair, PriceType.MidPrice)
                amount = self.config.total_amount_quote / price / Decimal(self.config.max_executors_per_side)
                trade_type = TradeType.BUY if signal > 0 else TradeType.SELL
                print(f"  Creating {trade_type} executor - Price: {price}, Amount: {amount}")
                
                return [CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=self.get_executor_config(trade_type, price, amount))]
            else:
                print(f"  Not creating executor - can_create_executor returned False")
        else:
            if signal == "KEY_MISSING":
                print(f"  Signal key missing from processed_data")
            else:
                print(f"  Signal is 0, no action needed")
        
        return []

    def can_create_executor(self, signal: int) -> bool:
        """
        Override with debugging for cooldown and executor conditions.
        """
        active_executors_by_signal_side = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and (x.side == TradeType.BUY if signal > 0 else TradeType.SELL))
        
        max_timestamp = max([executor.timestamp for executor in active_executors_by_signal_side], default=0)
        current_time = self.market_data_provider.time()
        time_since_last = current_time - max_timestamp
        
        active_executors_condition = len(active_executors_by_signal_side) < self.config.max_executors_per_side
        cooldown_condition = time_since_last > self.config.cooldown_time
        
        print(f"    Active executors: {len(active_executors_by_signal_side)}/{self.config.max_executors_per_side}")
        print(f"    Time since last: {time_since_last}s > {self.config.cooldown_time}s cooldown")
        print(f"    Conditions - Active: {active_executors_condition}, Cooldown: {cooldown_condition}")
        
        return active_executors_condition and cooldown_condition

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        Override to debug if this method is called at all.
        """
        if not hasattr(self, '_determine_call_count'):
            self._determine_call_count = 0
        self._determine_call_count += 1
        
        signal = self.processed_data.get("signal", "MISSING")
        timestamp = self.market_data_provider.time()
        current_time = pd.to_datetime(timestamp, unit='s')
        
        print(f"[DEMA SuperTrend] determine_executor_actions call #{self._determine_call_count}")
        print(f"  Current time: {current_time} (TS: {timestamp})")
        print(f"  Signal: {signal}")
        print(f"  processed_data keys: {list(self.processed_data.keys())}")
        
        actions = super().determine_executor_actions()
        print(f"  Returned {len(actions)} actions")
        return actions

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        """
        Stop actions based on SuperTrend reversal.
        """
        stop_actions = []
        
        # Get current SuperTrend direction
        current_supertrend_direction = self.processed_data.get("current_supertrend_direction")
        
        if current_supertrend_direction is None:
            return stop_actions
        
        # Get active executors
        active_executors = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active
        )
        
        for executor in active_executors:
            # Close long positions when SuperTrend turns red
            if executor.side == TradeType.BUY and current_supertrend_direction == -1:
                stop_actions.append(StopExecutorAction(
                    controller_id=self.config.id,
                    executor_id=executor.id
                ))
            
            # Close short positions when SuperTrend turns green
            elif executor.side == TradeType.SELL and current_supertrend_direction == 1:
                stop_actions.append(StopExecutorAction(
                    controller_id=self.config.id,
                    executor_id=executor.id
                ))
        
        return stop_actions