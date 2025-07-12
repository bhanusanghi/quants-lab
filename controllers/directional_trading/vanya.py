from typing import List

import pandas_ta as ta  # noqa: F401
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)
from pydantic import Field, field_validator


class VanyaControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name: str = "vanya_controller"
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(default=None)
    candles_trading_pair: str = Field(default=None)
    interval: str = "1m" 
    ema_short: int = 8
    ema_medium: int = 29 
    ema_long: int = 31
    rsi_length: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    bb_length: int = 20
    bb_std: float = 2.0
    bb_z_score_period: int = 100
    bb_z_threshold: float = 1.5
    up_streak_pct: float = 0.003
    triple_tol_pct: float = 0.002
    triple_min_spacing: int = 5

    @field_validator("candles_connector", mode="before")
    @classmethod
    def set_candles_connector(cls, v, values):
        if v is None or v == "":
            return values.get("connector_name")
        return v

    @field_validator("candles_trading_pair", mode="before")
    @classmethod
    def set_candles_trading_pair(cls, v, values):
        if v is None or v == "":
            return values.get("trading_pair")
        return v


class VanyaController(DirectionalTradingControllerBase):

    def __init__(self, config: VanyaControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = 1000
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records)

        # ─── Technical indicators ─────────────────────────────────────────────
        df.ta.ema(length=self.config.ema_short, append=True)
        df.ta.ema(length=self.config.ema_medium, append=True)
        df.ta.ema(length=self.config.ema_long, append=True)
        
        # try with sma to catch early?
        df.ta.sma(length=self.config.ema_short, append=True)
        df.ta.sma(length=self.config.ema_medium, append=True)
        df.ta.sma(length=self.config.ema_long, append=True)
        df.ta.rsi(length=self.config.rsi_length, append=True)
        # Bollinger Bands regime filter
        df.ta.bbands(length=self.config.bb_length, std=self.config.bb_std, append=True)

        # ─── Momentum: 3‑bar streak (up OR down)  ────────────────────────────
        df["up_streak"] = (df["close"] > df["close"].shift(1)) & \
                          (df["close"].shift(1) > df["close"].shift(2)) & \
                          (df["close"].shift(2) > df["close"].shift(3))

        df["down_streak"] = (df["close"] < df["close"].shift(1)) & \
                            (df["close"].shift(1) < df["close"].shift(2)) & \
                            (df["close"].shift(2) < df["close"].shift(3))

        # % move over the same 3‑bar window
        df["mom3_pct"] = df["close"].pct_change(3)

        df["mom_signal_up"] = (df["up_streak"] & (df["mom3_pct"] > self.config.up_streak_pct)).astype(int)
        df["mom_signal_down"] = (df["down_streak"] & (df["mom3_pct"] < -self.config.up_streak_pct)).astype(int)

        # Combined signal: +1 for upward momentum, -1 for downward, 0 otherwise
        df["mom_signal"] = 0
        df.loc[df["mom_signal_up"] == 1, "mom_signal"] = 1
        df.loc[df["mom_signal_down"] == 1, "mom_signal"] = -1

        # Trend filter (EMA ribbon)
        short_ema  = df[f"EMA_{self.config.ema_short}"]
        medium_ema = df[f"EMA_{self.config.ema_medium}"]
        long_ema   = df[f"EMA_{self.config.ema_long}"]
        # short_ema  = df[f"SMA_{self.config.ema_short}"]
        # medium_ema = df[f"SMA_{self.config.ema_medium}"]
        # long_ema   = df[f"SMA_{self.config.ema_long}"]
        trend_up   = (short_ema > medium_ema) & (medium_ema > long_ema)

        # Volatility regime: Bollinger band‑width z‑score
        upper_col = f"BBU_{self.config.bb_length}_{self.config.bb_std}"
        lower_col = f"BBL_{self.config.bb_length}_{self.config.bb_std}"
        df["bb_width"] = df[upper_col] - df[lower_col]
        df["bb_width_z"] = (
            (df["bb_width"] - df["bb_width"].rolling(self.config.bb_z_score_period).mean())
            / df["bb_width"].rolling(self.config.bb_z_score_period).std()
        )
        bb_breakout = df["bb_width_z"] > self.config.bb_z_threshold

        # RSI values
        rsi = df[f"RSI_{self.config.rsi_length}"]
        
        # ─── Triple‑top / Triple‑bottom detection ────────────────────────────
        # Pivot highs / lows using simple rolling windows (no pandas_ta internals)
        left = right = 3
        window = left + right + 1
        
        # Pivot high: highest in window and strictly greater than at least one neighbour
        df["piv_high"] = (
            (df["high"] == df["high"].rolling(window, center=True).max()) &
            (df["high"] > df["high"].shift(1))
        ).astype(int)

        # Pivot low: lowest in window and strictly lower than at least one neighbour
        df["piv_low"] = (
            (df["low"] == df["low"].rolling(window, center=True).min()) &
            (df["low"] < df["low"].shift(1))
        ).astype(int)
 
        # Check last three pivot highs
        df["triple_top"] = 0
        df["triple_bottom"] = 0
        piv_h_idx = df.index[df["piv_high"] == 1]
        if len(piv_h_idx) >= 3:
            last3h = piv_h_idx[-3:]
            highs = df.loc[last3h, "high"]
            level_ok = (highs.max() - highs.min()) / highs.mean() <= self.config.triple_tol_pct
            spacing_ok = (last3h.to_series().diff().min() >= self.config.triple_min_spacing)
            if level_ok and spacing_ok:
                df.at[df.index[-1], "triple_top"] = 1
 
        # Check last three pivot lows
        piv_l_idx = df.index[df["piv_low"] == 1]
        if len(piv_l_idx) >= 3:
            last3l = piv_l_idx[-3:]
            lows = df.loc[last3l, "low"]
            level_ok = (lows.max() - lows.min()) / lows.mean() <= self.config.triple_tol_pct
            spacing_ok = (last3l.to_series().diff().min() >= self.config.triple_min_spacing)
            if level_ok and spacing_ok:
                df.at[df.index[-1], "triple_bottom"] = 1

        # ─── Composite directional signal ────────────────────────────────────
        long_condition = trend_up & (rsi < self.config.rsi_overbought) & \
                         (df["mom_signal"] == 1) & bb_breakout & \
                         (~df["triple_top"].astype(bool))
 
        short_condition = (~trend_up) & (rsi > self.config.rsi_oversold) & \
                          (df["mom_signal"] == -1) & bb_breakout & \
                          (~df["triple_bottom"].astype(bool))

        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1


        self.processed_data["signal"] = int(df["signal"].iloc[-1])
        self.processed_data["features"] = df