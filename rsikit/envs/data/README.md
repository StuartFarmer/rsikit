# BTC training data attribution

`bitcoin_train.csv` is derived from Coin Metrics Community API `PriceUSD` daily
data: https://community-api.coinmetrics.io/v4/timeseries/asset-metrics

Attribution: Coin Metrics, https://coinmetrics.io/ and https://github.com/coinmetrics/data.
Data license: CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/.
The original data was reduced to date and USD price and split chronologically.
Downloaded September 19, 2026; training dates: 2016-09-19 through 2023-09-19.
Full provenance is in the repository's `data/bitcoin/metadata.json`.
The validation panel is deliberately not included in the installed package.
