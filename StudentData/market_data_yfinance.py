import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


class MarketDataCollector:
    def __init__(self):
        # Define all ticker categories
        self.tickers = {
            'defi_tokens': {
                'MKR-USD': 'Maker',
                'SNX-USD': 'Synthetix',
                'REN-USD': 'Ren',
                'QNT-USD': 'Quant',
            },
            'layer1_tokens': {
                'BTC-USD': 'Bitcoin',
                'ETH-USD': 'Ethereum',
                'ADA-USD': 'Cardano',
                'XRP-USD': 'XRP',
                # 'BCH-USD': 'Bitcoin Cash',
                # 'LTC-USD': 'Litecoin',
                'XMR-USD': 'Monero'
            },
            'layer2_tokens': {
                'MATIC-USD': 'Polygon',
            },
            'stablecoins': {
                'USDT-USD': 'Tether',
                'USDC-USD': 'USD Coin',
                'DAI-USD': 'DAI',
                'BUSD-USD': 'Binance USD',
                'TUSD-USD': 'TrueUSD',
            },
            'us_indices': {
                # '^DJI': 'Dow Jones',
                '^IXIC': 'NASDAQ Composite',
                '^GSPC': 'S&P 500',
                '^VIX': 'VIX'
            },
            'global_indices': {
                '^FTSE': 'FTSE 100',
                # '^GDAXI': 'DAX',
                # '^FCHI': 'CAC 40',
                # 'FTSEMIB.MI': 'FTSE MIB',
                # '^IBEX': 'IBEX 35',
                '^STOXX50E': 'STOXX Europe 50',
                '^N225': 'Nikkei 225',
                '^HSI': 'Hang Seng',
                '000001.SS': 'Shanghai Composite',
                '^BSESN': 'BSE SENSEX',
                '^STI': 'Straits Times',
                '^AXJO': 'S&P/ASX 200'
            },
            'forex': {
                'EUR=X': 'EUR/USD',
                # 'JPY=X': 'JPY/USD',
                # 'GBP=X': 'GBP/USD',
                # 'AUD=X': 'AUD/USD',
            },
            'commodities': {
                'GC=F': 'Gold',
                'CL=F': 'Crude Oil WTI',
                'SI=F': 'Silver'
            },
            'bonds': {
                '^TNX': 'US 10Y Treasury',
                # '^IRX': 'US 13-Week Treasury Bill',
                # '^FVX': 'US 5Y Treasury',
                # '^TYX': 'US 30Y Treasury',
            },
            'futures': {
                # 'NQ=F': 'E-Mini NASDAQ',
                # 'YM=F': 'E-Mini Dow',
                # 'ES=F': 'E-Mini S&P 500'
            },
            'defi_infrastructure': {
                'LINK-USD': 'Chainlink',
                # 'BAND-USD': 'Band Protocol',
                'TRB-USD': 'Tellor',
            },
            'liquid_staking': {
                'RPL-USD': 'Rocket Pool',
                'ANKR-USD': 'Ankr',
            },
            'cross_chain': {
                'RUNE-USD': 'THORChain',
                'ATOM-USD': 'Cosmos',
            }
        }

    def collect_market_data(self, start_date, end_date):
        fetch_date = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=380)).strftime('%Y-%m-%d')
        full_date_range = pd.date_range(start=fetch_date, end=end_date, freq='D')
        all_data = []

        for category, tickers in self.tickers.items():
            for ticker, name in tickers.items():
                print(f"Fetching data for {name} ({ticker})")
                ticker_data = yf.download(ticker, start=fetch_date, end=end_date, interval='1d')

                ticker_data.index = ticker_data.index.tz_convert(None)
                ticker_data = ticker_data.reindex(full_date_range)

                # Rename columns for uniqueness
                ticker_data.columns = [
                    ''.join([word.lower() if i == 0 else word.capitalize() for i, word in enumerate(col[0].split())]) +
                    col[1]
                    for col in ticker_data.columns
                ]

                # drop adjusted as it is highly correlated
                ticker_data = ticker_data.drop(columns=[f"adjClose{ticker}"])

                volume_col = f'volume{ticker}'
                non_volume_cols = [col for col in ticker_data.columns if col != volume_col]

                # Forward fill price and other non-volume data
                ticker_data[non_volume_cols] = ticker_data[non_volume_cols].ffill()

                # Fill volume with 0
                if volume_col in ticker_data.columns:
                    ticker_data[volume_col] = ticker_data[volume_col].fillna(0)

                # Basic price metrics
                ticker_data[f'priceVolatilityWeekly{ticker}'] = ticker_data[f'close{ticker}'].rolling(window=7).std()
                ticker_data[f'priceVolatilityMonthly{ticker}'] = ticker_data[f'close{ticker}'].rolling(window=30).std()
                ticker_data[f'volatilityRatio{ticker}'] = ticker_data[f'priceVolatilityWeekly{ticker}'] / ticker_data[
                    f'priceVolatilityMonthly{ticker}']

                # Price changes at various intervals
                for period, days in [('Daily', 1), ('Weekly', 7), ('Monthly', 30), ('Quarterly', 90), ('Yearly', 365)]:
                    ticker_data[f'pctChange{period}{ticker}'] = ticker_data[f'close{ticker}'].pct_change(periods=days)

                # Technical indicators
                ticker_data[f'rsi{ticker}'] = self._calculate_rsi(ticker_data[f'close{ticker}'])
                ticker_data[f'closeAvgWeekly{ticker}'] = ticker_data[f'close{ticker}'].rolling(window=7).mean()
                ticker_data[f'closeAvgMonthly{ticker}'] = ticker_data[f'close{ticker}'].rolling(window=30).mean()

                # Market efficiency metrics
                ticker_data[f'marketEfficiency{ticker}'] = abs(
                    1 - (ticker_data[f'close{ticker}'] / ticker_data[f'closeAvgMonthly{ticker}'])
                ).rolling(7).mean()

                # Drop volume column if its sum is zero
                volume_col_name = f"volume{ticker}"
                if volume_col_name in ticker_data.columns and ticker_data[volume_col_name].sum() == 0:
                    ticker_data = ticker_data.drop(columns=[volume_col_name])
                else:
                    ticker_data[f'volumeAvgWeekly{ticker}'] = ticker_data[f'volume{ticker}'].rolling(window=7).mean()
                    ticker_data[f'marketDepth{ticker}'] = ticker_data[f'volume{ticker}'] / ticker_data[
                        f'priceVolatilityWeekly{ticker}']

                    # redundant for stable
                    if category != "stablecoins":
                        ticker_data[f'marketCap{ticker}'] = ticker_data[f'close{ticker}'] * ticker_data[f'volume{ticker}']
                        ticker_data[f'marketCapAvgWeekly{ticker}'] = ticker_data[f'marketCap{ticker}'].rolling(
                            window=7).mean()
                        ticker_data[f'liquidityRatio{ticker}'] = ticker_data[f'volume{ticker}'] / ticker_data[
                            f'marketCap{ticker}']

                    # Liquidity metrics
                    ticker_data[f'amihudIlliquidity{ticker}'] = abs(ticker_data[f'pctChangeDaily{ticker}']) / (
                            ticker_data[f'volume{ticker}'] * ticker_data[f'close{ticker}'])

                    # Liquidity stress indicators
                    ticker_data[f'liquidityStress{ticker}'] = (
                            ticker_data[f'amihudIlliquidity{ticker}'].rolling(7, min_periods=1).mean() /
                            ticker_data[f'amihudIlliquidity{ticker}'].rolling(30, min_periods=1).mean()
                    )

                    # Market sentiment indicators
                    ticker_data[f'bullishDivergence{ticker}'] = (
                            (ticker_data[f'close{ticker}'] < ticker_data[f'close{ticker}'].shift(1)) &
                            (ticker_data[f'volume{ticker}'] > ticker_data[f'volume{ticker}'].shift(1))
                    ).astype(int)

                ticker_data = ticker_data.ffill()

                all_data.append(ticker_data)

        combined_data = pd.concat(all_data, axis=1)
        combined_data = combined_data[combined_data.index >= start_date]

        # Add timestamp column for joining
        # with warnings.catch_warnings():  # suppress the PerformanceWarning from the dataframe being fragmented
        #     warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
        #     combined_data['timestamp'] = combined_data.index.astype('int64') // 10 ** 9

        # combined_data = combined_data[['timestamp'] + [col for col in combined_data.columns if col != 'timestamp']]

        return combined_data

    def _calculate_rsi(self, prices, periods=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


def main():
    collector = MarketDataCollector()

    start_date = '2020-11-28'
    end_date   = '2024-05-25'

    data = collector.collect_market_data(start_date, end_date)

    output_file = f'market_data_{start_date}_to_{end_date}.csv'
    data.to_csv(output_file)


if __name__ == "__main__":
    main()
