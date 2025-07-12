from decimal import Decimal
from typing import List, Optional

import pandas_ta as ta
from pydantic import Field, field_validator

from hummingbot.core.data_type.common import MarketDict, OrderType, PositionMode, PriceType, TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction


class MACDBBRSIMultiTokenControllerConfig(ControllerConfigBase):
    controller_name: str = "macd_bb_rsi_multi_token"
    
    # Exchange configuration
    exchange: str = Field(default="hyperliquid_perpetual")
    trading_pairs: List[str] = Field(default=["ETH-USD"])
    candles_exchange: str = Field(default="binance_perpetual")
    candles_pairs: List[str] = Field(default=["ETH-USDT"])
    candles_interval: str = Field(default="5m")
    candles_length: int = Field(default=60, gt=0)
    
    # RSI Configuration
    rsi_low: float = Field(default=20, gt=0)
    rsi_high: float = Field(default=80, gt=0)
    rsi_length: int = Field(default=14, gt=0)
    
    # MACD Configuration
    macd_fast: int = Field(default=6, gt=0)
    macd_slow: int = Field(default=13, gt=0)
    macd_signal: int = Field(default=4, gt=0)
    
    # Bollinger Bands Configuration
    bb_length: int = Field(default=20, gt=0)
    bb_std: float = Field(default=2.2, gt=0)
    
    # Bollinger Bands Threshold Configuration
    bb_long_threshold: float = Field(default=0.0)
    bb_short_threshold: float = Field(default=1.0)
    
    # Order Configuration
    order_amount_quote: Decimal = Field(default=30, gt=0)
    leverage: int = Field(default=10, gt=0)
    position_mode: PositionMode = Field(default="ONEWAY")

    # Triple Barrier Configuration
    stop_loss: Decimal = Field(default=Decimal("0.03"), gt=0)
    take_profit: Decimal = Field(default=Decimal("0.01"), gt=0)
    time_limit: int = Field(default=60 * 15, gt=0)
    
    # Backtesting compatibility - use first trading pair and candles pair
    @property
    def connector_name(self) -> str:
        return self.exchange
    
    @property
    def trading_pair(self) -> str:
        return self.trading_pairs[0] if self.trading_pairs else "ETH-USD"
    
    @property
    def total_amount_quote(self) -> Decimal:
        return self.order_amount_quote
    
    def get_controller_class(self):
        return MACDBBRSIMultiTokenController

    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        return TripleBarrierConfig(
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            time_limit=self.time_limit,
            open_order_type=OrderType.MARKET,
            take_profit_order_type=OrderType.LIMIT,
            stop_loss_order_type=OrderType.MARKET,
            time_limit_order_type=OrderType.MARKET
        )

    @field_validator('position_mode', mode="before")
    @classmethod
    def validate_position_mode(cls, v: str) -> PositionMode:
        if v.upper() in PositionMode.__members__:
            return PositionMode[v.upper()]
        raise ValueError(f"Invalid position mode: {v}. Valid options are: {', '.join(PositionMode.__members__)}")
    
    @field_validator('trading_pairs', mode="before")
    @classmethod
    def validate_trading_pairs(cls, v) -> List[str]:
        if isinstance(v, str):
            return [pair.strip() for pair in v.split(',')]
        return v
    
    @field_validator('candles_pairs', mode="before")
    @classmethod
    def validate_candles_pairs(cls, v) -> List[str]:
        if isinstance(v, str):
            return [pair.strip() for pair in v.split(',')]
        return v
    
    def update_markets(self, markets: MarketDict) -> MarketDict:
        """
        Update markets to include both trading and candles exchanges.
        """
        # Add trading exchange and pairs
        if self.exchange not in markets:
            markets[self.exchange] = set()
        markets[self.exchange].update(self.trading_pairs)
        
        # Add candles exchange and pairs
        if self.candles_exchange not in markets:
            markets[self.candles_exchange] = set()
        markets[self.candles_exchange].update(self.candles_pairs)
        
        return markets


class MACDBBRSIMultiTokenController(ControllerBase):
    """
    Controller using RSI, MACD, and Bollinger Bands to generate trading signals.
    Fetches candles from one market and executes trades on another.
    """
    
    def __init__(self, config: MACDBBRSIMultiTokenControllerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.max_records = max(config.macd_slow, config.macd_fast, config.macd_signal, config.bb_length, config.candles_length) + 20
        
        # Initialize candles config if not set
        if len(config.candles_config) == 0:
            for candles_pair in config.candles_pairs:
                config.candles_config.append(CandlesConfig(
                    connector=config.candles_exchange,
                    trading_pair=candles_pair,
                    interval=config.candles_interval,
                    max_records=self.max_records
                ))
        
        # Store indicators per trading pair
        self.current_rsi = {}
        self.current_macd = {}
        self.current_macd_histogram = {}
        self.current_bbp = {}
        self.current_signal = {}
        self.account_config_set = False

    async def update_processed_data(self):
        """
        Update processed data by calculating indicators for each candles pair.
        """
        for candles_pair in self.config.candles_pairs:
            try:
                candles = self.market_data_provider.get_candles_df(
                    self.config.candles_exchange,
                    candles_pair,
                    self.config.candles_interval,
                    self.max_records
                )
                
                if candles is not None and not candles.empty:
                    # Calculate indicators
                    candles.ta.rsi(length=self.config.rsi_length, append=True)
                    candles.ta.bbands(length=self.config.bb_length, std=self.config.bb_std, append=True)
                    candles.ta.macd(fast=self.config.macd_fast, slow=self.config.macd_slow, signal=self.config.macd_signal, append=True)
                    
                    # Get current indicator values and store per trading pair
                    self.current_rsi[candles_pair] = candles.iloc[-1][f"RSI_{self.config.rsi_length}"]
                    self.current_bbp[candles_pair] = candles.iloc[-1][f"BBP_{self.config.bb_length}_{self.config.bb_std}"]
                    self.current_macd[candles_pair] = candles.iloc[-1][f"MACD_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
                    self.current_macd_histogram[candles_pair] = candles.iloc[-1][f"MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
                    
                    # Calculate signal
                    signal = self._calculate_signal(candles, candles_pair)
                    self.current_signal[candles_pair] = signal
                    
                    # Store processed data for backtesting compatibility
                    if not hasattr(self, 'processed_data'):
                        self.processed_data = {}
                    self.processed_data["features"] = candles
                    self.processed_data["signal"] = signal
                    
            except Exception as e:
                self.logger().error(f"Error updating processed data for {candles_pair}: {e}")
                
    def _calculate_signal(self, candles, candles_pair: str) -> Optional[float]:
        """
        Calculate trading signal based on RSI, MACD, and Bollinger Bands.
        """
        try:
            # Define combined signal conditions
            rsi_condition = candles[f"RSI_{self.config.rsi_length}"]
            bbp_condition = candles[f"BBP_{self.config.bb_length}_{self.config.bb_std}"]
            macd_condition = candles[f"MACD_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
            macdh_condition = candles[f"MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
            
            # Generate combined signals
            long_condition = (rsi_condition < self.config.rsi_low) & (bbp_condition < self.config.bb_long_threshold) & (macdh_condition > 0) & (macd_condition < 0)
            short_condition = (rsi_condition > self.config.rsi_high) & (bbp_condition > self.config.bb_short_threshold) & (macdh_condition < 0) & (macd_condition > 0)
            
            candles["signal"] = 0
            candles.loc[long_condition, "signal"] = 1
            candles.loc[short_condition, "signal"] = -1
            
            return candles.iloc[-1]["signal"] if not candles.empty else None
            
        except Exception as e:
            self.logger().error(f"Error calculating signal for {candles_pair}: {e}")
            return None

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        Determine what executor actions to take based on signals.
        """
        actions = []
        
        # Apply initial settings if not done
        if not self.account_config_set:
            self._apply_initial_settings()
        
        # Check signals for each trading pair
        for i, trading_pair in enumerate(self.config.trading_pairs):
            if i >= len(self.config.candles_pairs):
                self.logger().warning(f"Index {i} out of range for candles_pairs. trading_pairs: {self.config.trading_pairs}, candles_pairs: {self.config.candles_pairs}")
                continue
                
            candles_pair = self.config.candles_pairs[i]
            signal = self.current_signal.get(candles_pair)
            
            if signal is not None:
                active_longs, active_shorts = self._get_active_executors_by_side(trading_pair)
                
                try:
                    mid_price = self.market_data_provider.get_price_by_type(
                        self.config.exchange,
                        trading_pair,
                        PriceType.MidPrice
                    )
                    
                    # Create new positions
                    if signal == 1 and len(active_longs) == 0:
                        actions.append(CreateExecutorAction(
                            executor_config=PositionExecutorConfig(
                                timestamp=self.market_data_provider.get_time(),
                                connector_name=self.config.exchange,
                                trading_pair=trading_pair,
                                side=TradeType.BUY,
                                entry_price=mid_price,
                                amount=self.config.order_amount_quote / mid_price,
                                triple_barrier_config=self.config.triple_barrier_config,
                                leverage=self.config.leverage
                            )))
                    elif signal == -1 and len(active_shorts) == 0:
                        actions.append(CreateExecutorAction(
                            executor_config=PositionExecutorConfig(
                                timestamp=self.market_data_provider.get_time(),
                                connector_name=self.config.exchange,
                                trading_pair=trading_pair,
                                side=TradeType.SELL,
                                entry_price=mid_price,
                                amount=self.config.order_amount_quote / mid_price,
                                triple_barrier_config=self.config.triple_barrier_config,
                                leverage=self.config.leverage
                            )))
                    
                    # Stop opposite positions
                    if signal == -1 and len(active_longs) > 0:
                        actions.extend([StopExecutorAction(executor_id=e.id) for e in active_longs])
                    elif signal == 1 and len(active_shorts) > 0:
                        actions.extend([StopExecutorAction(executor_id=e.id) for e in active_shorts])
                        
                except Exception as e:
                    self.logger().error(f"Error determining executor actions for {trading_pair}: {e}")
                    
        return actions

    def _get_active_executors_by_side(self, trading_pair: str):
        """
        Get active executors by side for a specific trading pair.
        """
        active_executors_by_trading_pair = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda e: e.connector_name == self.config.exchange and e.trading_pair == trading_pair and e.is_active
        )
        active_longs = [e for e in active_executors_by_trading_pair if e.side == TradeType.BUY]
        active_shorts = [e for e in active_executors_by_trading_pair if e.side == TradeType.SELL]
        return active_longs, active_shorts

    def _apply_initial_settings(self):
        """
        Apply initial settings like position mode and leverage.
        """
        if not self.account_config_set:
            try:
                # Note: In controller architecture, we can't directly access connectors
                # This would typically be handled by the strategy orchestrator
                self.logger().info("Initial settings would be applied by strategy orchestrator")
                self.account_config_set = True
            except Exception as e:
                self.logger().error(f"Error applying initial settings: {e}")

    def to_format_status(self) -> List[str]:
        """
        Format status for UI display.
        """
        lines = []
        
        # Create compact trading pairs overview
        lines.extend(["Trading Pairs Overview:"])
        
        # Header for the grid
        header = f"{'Pair':<12} {'RSI':<7} {'MACD':<9} {'MACDH':<9} {'BBP':<7} {'Signal':<7} {'Long':<4} {'Short':<5}"
        lines.append(header)
        lines.append('-' * len(header))

        # Display each trading pair in compact format
        for i, candles_pair in enumerate(self.config.candles_pairs):
            # Get indicator values for this pair
            rsi_val = self.current_rsi.get(candles_pair, -1)
            macd_val = self.current_macd.get(candles_pair, -1)
            macdh_val = self.current_macd_histogram.get(candles_pair, -1)
            bbp_val = self.current_bbp.get(candles_pair, -1)
            signal = self.current_signal.get(candles_pair, -1)
            
            # Format signal display
            signal_text = "LONG" if signal == 1 else "SHORT" if signal == -1 else "NONE"
            
            # Get active positions for this pair
            if i < len(self.config.trading_pairs):
                actual_trading_pair = self.config.trading_pairs[i]
                active_longs, active_shorts = self._get_active_executors_by_side(actual_trading_pair)
            else:
                active_longs, active_shorts = [], []
            
            # Format the row
            pair_display = candles_pair.replace('-', '/') if len(candles_pair) > 12 else candles_pair
            row = f"{pair_display:<12} {rsi_val:>6.1f} {macd_val:>8.4f} {macdh_val:>8.4f} {bbp_val:>6.3f} {signal_text:>7} {len(active_longs):>4} {len(active_shorts):>5}"
            lines.append(row)

        # Add threshold info
        lines.extend([
            "",
            f"Thresholds: RSI({self.config.rsi_low}-{self.config.rsi_high}) | BB({self.config.bb_long_threshold:.3f}-{self.config.bb_short_threshold:.3f})"
        ])

        return lines