## Project Overview

QuantsLab is a Python-based quantitative trading research framework built on top of Hummingbot. It provides tools for backtesting, optimization, data collection, and deployment of trading strategies across multiple market types (CLOB, AMM, DeFi).

## Development Setup

### Environment Management
- Use Conda for dependency management: `conda activate quants-lab`
- Install dependencies: `make install`
- Environment defined in `environment.yml` with Python 3.12

### Code Style
- Black formatting with 130 character line length
- isort for import sorting
- Pre-commit hooks configured for code quality
- Format code: `black .` and `isort .`

## Core Architecture

### Task-Based System
- **BaseTask**: Abstract base class for all tasks in `core/task_base.py`
- **TaskRunner**: Orchestrates task execution from YAML configs in `core/task_runner.py`
- **TaskOrchestrator**: Manages concurrent task execution with frequency-based scheduling
- Execute tasks: `python run_tasks.py --config config/tasks.yml`

### Controllers (Trading Strategies)
- **DirectionalTradingControllerBase**: Base class for trend-following strategies
- **MarketMakingControllerBase**: Base class for market-making strategies
- Controllers located in `controllers/` organized by strategy type:
  - `directional_trading/`: Trend-following strategies (Bollinger, MACD, RSI, etc.)
  - `market_making/`: Market-making strategies (PMM, Dynamic MM)
  - `generic/`: General strategies (Grid, StatArb)

### Data Architecture
- **Data Sources**: Multi-exchange support via `core/data_sources/`
  - CLOB: Centralized exchanges (Binance, etc.)
  - AMM: Decentralized exchanges (Solana, etc.)
  - External APIs: CoinGecko, GeckoTerminal, Dune Analytics
- **Data Structures**: Standardized data models in `core/data_structures/`
- **Backtesting**: Historical data processing and strategy simulation

### Core Components
- **Backtesting Engine**: `core/backtesting/engine.py` - Strategy simulation framework
- **Features**: Technical indicators and market features in `core/features/`
- **Performance**: Analytics and reporting in `core/performance/`
- **Services**: External API clients in `core/services/`

## Common Development Tasks

### Running Tasks
```bash
# Build Docker image
make build

# Start databases (MongoDB, PostgreSQL)
make run-db

# Run task with specific config
make run-task config=tasks.yml

# Stop services
make stop-task
make stop-db
```

### Strategy Development
1. Create controller class inheriting from appropriate base class
2. Implement strategy logic in controller methods
3. Create corresponding research notebook in `research_notebooks/`
4. Use backtesting engine for strategy validation
5. Create optimization notebooks for parameter tuning

### Backtesting Workflow
1. Load historical data via data sources
2. Configure strategy parameters
3. Run backtest using `BacktestingEngine`
4. Analyze results with performance metrics
5. Optimize parameters using Optuna framework

### Database Access
- MongoDB: Market data, pool information, strategy results
- PostgreSQL/TimescaleDB: Time-series data storage
- Connection configs in task YAML files or environment variables

## Key Directories

- `controllers/`: Trading strategy implementations
- `core/`: Core framework components (backtesting, data, services)
- `tasks/`: Automated data collection and processing tasks
- `research_notebooks/`: Jupyter notebooks for strategy research
- `config/`: YAML configuration files for tasks and deployments
- `data/`: Local data storage (candles, backtest results)

## Configuration Management

### Task Configuration
- YAML-based task configuration in `config/` directory
- Environment variables for sensitive data (database credentials)
- Task scheduling via frequency-based execution

### Strategy Configuration
- Pydantic models for type-safe configuration
- JSON schema validation for parameters
- Environment-specific config files

## Testing and Validation

### Research Notebooks
- Each strategy has dedicated research notebooks
- Standard workflow: EDA → Strategy Design → Backtesting → Optimization
- Notebooks serve as documentation and validation

### Backtesting Validation
- Historical data validation before backtesting
- Performance metrics calculation and reporting
- Risk management and drawdown analysis

## Data Management

### Supported Data Sources
- **CLOB**: Price, orderbook, trades, funding rates
- **AMM**: Liquidity, pool stats, swap data
- **Market Data**: OHLCV, volume, market stats
- **External**: Token metrics, exchange data

### Data Storage
- Parquet files for historical candle data
- Database storage for real-time and processed data
- Caching mechanisms for performance optimization

## Deployment

### Docker Support
- Dockerfile for containerized deployment
- Docker Compose for multi-service orchestration
- Separate configs for development and production

### Task Deployment
- YAML-based task scheduling
- Environment variable configuration
- Monitoring and logging integration

## Project Environment

- Python packages are installed through conda environment in this project
- Package location: `/opt/homebrew/Caskroom/miniconda/base/envs/quants-lab/lib/python3.12/site-packages/`