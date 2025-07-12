from typing import List
import numpy as np
import pandas as pd

import pandas_ta as ta  # noqa: F401
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)


class VolatilityCompressionControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "macd_bb_v1"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the connector for the candles data, leave empty to use the same exchange as the connector: ",
            "prompt_on_new": True})
    trading_pair: str = Field(
        default=None,
        json_schema_extra={
            "prompt": "Enter the trading pair for the candles data, leave empty to use the same trading pair as the connector: ",
            "prompt_on_new": True})
    interval: str = Field(
        default="3m",
        json_schema_extra={
            "prompt": "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            "prompt_on_new": True})
    
    # Original BB parameters (keeping for backward compatibility)
    bb_length: int = Field(
        default=100,
        json_schema_extra={"prompt": "Enter the Bollinger Bands length: ", "prompt_on_new": True})
    bb_std: float = Field(default=2.0)
    bb_long_threshold: float = Field(default=0.0)
    bb_short_threshold: float = Field(default=1.0)
    
    # New BB parameters for squeeze detection
    bb_squeeze_length: int = Field(default=20)
    bb_squeeze_std: float = Field(default=2.0)
    bb_squeeze_lookback: int = Field(default=100)
    
    # MACD parameters
    macd_fast: int = Field(
        default=21,
        json_schema_extra={"prompt": "Enter the MACD fast period: ", "prompt_on_new": True})
    macd_slow: int = Field(
        default=42,
        json_schema_extra={"prompt": "Enter the MACD slow period: ", "prompt_on_new": True})
    macd_signal: int = Field(
        default=9,
        json_schema_extra={"prompt": "Enter the MACD signal period: ", "prompt_on_new": True})
    
    # RSI parameters (original and new)
    rsi_low: float = Field(default=20, gt=0)
    rsi_high: float = Field(default=80, gt=0)
    rsi_length: int = Field(default=14, gt=0)
    rsi_short_length: int = Field(default=7, gt=0)  # New RSI(7)
    
    # Keltner Channel parameters
    kc_length: int = Field(default=20)
    kc_scalar: float = Field(default=1.5)
    
    # StochRSI parameters
    stochrsi_length: int = Field(default=7)
    stochrsi_rsi_length: int = Field(default=7)
    stochrsi_k: int = Field(default=3)
    stochrsi_d: int = Field(default=3)
    stochrsi_low: float = Field(default=5)
    stochrsi_high: float = Field(default=95)
    
    # OBV divergence parameters
    obv_slope_window: int = Field(default=20)
    divergence_threshold: float = Field(default=0.5)

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("connector_name")
        return v

    @field_validator("trading_pair", mode="before")
    @classmethod
    def set_trading_pair(cls, v, validation_info: ValidationInfo):
        if v is None or v == "":
            return validation_info.data.get("trading_pair")
        return v


class VolatilityCompressionController(DirectionalTradingControllerBase):

    def __init__(self, config: VolatilityCompressionControllerConfig, *args, **kwargs):
        self.config = config
        # Calculate max records needed
        self.max_records = max(
            config.macd_slow, 
            config.macd_fast, 
            config.macd_signal, 
            config.bb_length,
            config.bb_squeeze_length + config.bb_squeeze_lookback,
            config.kc_length,
            config.obv_slope_window + 50,  # Extra buffer for OBV calculations
            config.stochrsi_length + config.stochrsi_rsi_length
        ) + 20
        
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    def calculate_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate slope using linear regression over a rolling window"""
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(window)
                if not np.isnan(y).any():
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.trading_pair,
            interval=self.config.interval,
            max_records=self.max_records
        )
        
        # Original indicators
        df.ta.bbands(length=self.config.bb_length, std=self.config.bb_std, append=True)
        df.ta.macd(fast=self.config.macd_fast, slow=self.config.macd_slow, 
                   signal=self.config.macd_signal, append=True)
        df.ta.rsi(length=self.config.rsi_length, append=True)
        
        # New indicators
        # 1. RSI(7)
        df.ta.rsi(length=self.config.rsi_short_length, append=True)
        
        # 2. Keltner Channels
        kc_result = df.ta.kc(length=self.config.kc_length, scalar=self.config.kc_scalar, append=True)
        
        # 3. Bollinger Bands for squeeze detection
        bb_squeeze = df.ta.bbands(length=self.config.bb_squeeze_length, std=self.config.bb_squeeze_std, append=True)
        
        # 4. BB Squeeze detection
        bb_width_col = f"BBB_{self.config.bb_squeeze_length}_{self.config.bb_squeeze_std}"
        df['bb_width_pct25'] = df[bb_width_col].rolling(window=self.config.bb_squeeze_lookback).quantile(0.25)
        df['bb_squeeze'] = df[bb_width_col] < df['bb_width_pct25']
        
        # 5. StochRSI
        stochrsi_result = df.ta.stochrsi(
            length=self.config.stochrsi_length, 
            rsi_length=self.config.stochrsi_rsi_length,
            k=self.config.stochrsi_k, 
            d=self.config.stochrsi_d, 
            append=True
        )
        
        # 6. OBV and divergence
        df['obv'] = df.ta.obv()
        df['price_slope'] = self.calculate_slope(df['close'], self.config.obv_slope_window)
        df['obv_slope'] = self.calculate_slope(df['obv'], self.config.obv_slope_window)
        
        # Normalize slopes
        price_std = df['close'].rolling(self.config.obv_slope_window).std()
        obv_std = df['obv'].rolling(self.config.obv_slope_window).std()
        df['price_slope_norm'] = df['price_slope'] / price_std.where(price_std > 0, 1)
        df['obv_slope_norm'] = df['obv_slope'] / obv_std.where(obv_std > 0, 1)
        
        # Divergence detection
        df['bullish_divergence'] = (
            (df['price_slope_norm'] < -self.config.divergence_threshold) & 
            (df['obv_slope_norm'] > self.config.divergence_threshold)
        )
        df['bearish_divergence'] = (
            (df['price_slope_norm'] > self.config.divergence_threshold) & 
            (df['obv_slope_norm'] < -self.config.divergence_threshold)
        )
        
        # Get indicator values
        bbp = df[f"BBP_{self.config.bb_length}_{self.config.bb_std}"]
        macdh = df[f"MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
        macd = df[f"MACD_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
        rsi = df[f"RSI_{self.config.rsi_length}"]
        rsi_short = df[f"RSI_{self.config.rsi_short_length}"]
        
        # Keltner Channels
        kc_lower = df[f"KCLe_{self.config.kc_length}_{self.config.kc_scalar}"]
        kc_upper = df[f"KCUe_{self.config.kc_length}_{self.config.kc_scalar}"]
        
        # StochRSI
        stochrsi_k = df[f"STOCHRSIk_{self.config.stochrsi_length}_{self.config.stochrsi_rsi_length}_{self.config.stochrsi_k}_{self.config.stochrsi_d}"]
        
        # Bollinger Bands for squeeze
        bb_lower_squeeze = df[f"BBL_{self.config.bb_squeeze_length}_{self.config.bb_squeeze_std}"]
        bb_upper_squeeze = df[f"BBU_{self.config.bb_squeeze_length}_{self.config.bb_squeeze_std}"]
        
        # Enhanced signal generation combining all indicators
        # Long conditions
        momentum_long = (
            ((rsi < self.config.rsi_low) | (rsi_short < 20)) |  # RSI oversold
            (stochrsi_k < self.config.stochrsi_low)  # StochRSI oversold
        )
        
        squeeze_long = df['bb_squeeze'] & (df['close'] <= kc_lower)  # Squeeze + price at KC lower
        
        original_long = (bbp < self.config.bb_long_threshold) & (macdh > 0) & (macd < 0)
        
        strong_long = df['bullish_divergence'] & (df['close'] <= bb_lower_squeeze)
        
        # Short conditions
        momentum_short = (
            ((rsi > self.config.rsi_high) | (rsi_short > 80)) |  # RSI overbought
            (stochrsi_k > self.config.stochrsi_high)  # StochRSI overbought
        )
        
        squeeze_short = df['bb_squeeze'] & (df['close'] >= kc_upper)  # Squeeze + price at KC upper
        
        original_short = (bbp > self.config.bb_short_threshold) & (macdh < 0) & (macd > 0)
        
        strong_short = df['bearish_divergence'] & (df['close'] >= bb_upper_squeeze)
        
        # Combine conditions
        long_condition = strong_long | (original_long & momentum_long) | (squeeze_long & momentum_long)
        short_condition = strong_short | (original_short & momentum_short) | (squeeze_short & momentum_short)
        
        # Generate signal
        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1
        
        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df
        
        # Add additional info for debugging/monitoring
        self.processed_data["indicators"] = {
            "rsi_7": rsi_short.iloc[-1],
            "rsi_14": rsi.iloc[-1],
            "stochrsi_k": stochrsi_k.iloc[-1],
            "bb_squeeze": df['bb_squeeze'].iloc[-1],
            "bullish_divergence": df['bullish_divergence'].iloc[-1],
            "bearish_divergence": df['bearish_divergence'].iloc[-1],
            "price_at_kc_lower": df['close'].iloc[-1] <= kc_lower.iloc[-1],
            "price_at_kc_upper": df['close'].iloc[-1] >= kc_upper.iloc[-1],
        }