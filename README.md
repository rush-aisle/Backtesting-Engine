# SPY Strategy Backtester: SMA Crossover, RSI Mean-Reversion, and ML Signal vs. Buy & Hold

This backtesting engine, created from scratch, compares two basic trading strategies (SMA crossover, RSI mean-reversion) as well as a logistic regression ML signal against passive buy-and-hold, on SPY daily prices from 2010–2026.

## Data

Used the `yfinance` library to extract SPY daily closing prices from 2010-01-01 to 2026-09-25

## Methodology

**Lookahead bias.** All 3 different signals mentioned above had to be shifted by one trading day before they were multiplied with the returns. This is due to the signal being determined from a day's closing price, meaning a change in signal can only be acted upon during the next trading day. Using it on the same day would not make sense, as it is making decisions on information that was not available. 

**Sharpe ratio.** Instead of using geometric compounding for the annual risk-free rate, the project converted the risk-free rate into a daily rate by just dividing it by the number of trading days per year (252). There is anyway a very small difference between the two approaches, so the simpler approach was implemented.

**Max drawdown.** This metric detects the largest percentage decline based on a running maximum. This is different from just calculating the difference between the maximum and minimum over the time period, since that ignores the order the peaks and troughs occur.

**Train/test split.** Split the dataset in half, using 2010-2019 for the training set, and 2020-2026 for the testing set. Splitting in half is better than shuffling the dates between the datasets since it would end up leaking future information when training, which will be a huge problem for ML models. It will also be difficult to calculate moving averages if the dates are all shuffled.

**Parameter selection.** The SMA crossover strategy has two parameters: a short moving average and a long moving average. Thus, you can have multiple different values satisfying these parameters. The RSI strategy has three parameters: window, entry point, and exit point. This project tests a small set of these parameter combinations specifically on the training period. The combinations with the highest metrics were then evaluated with the test period.
- SMA pairs tested: (5,20), (10,30), (20,50), (50,200)
- RSI (window, entry, exit) tested: (14,30,50), (14,25,60), (7,30,50), (21,30,70)

**ML signal.** Implemented a logistic regression model that used same-day features:
- MA spread ((ShortMA - LongMA) / LongMA)
- RSI value
- 10-day rolling volatility
- Same-day return
Using these features, it would aim to predict whether the *next* day has a positive return. Specifically, I used `class_weight='balanced'` (see Challenges below for why). The predictions were shifted one day forward like the other two strategies, and I calculated the same Sharpe ratio and max drawdown metrics.


## Results (test period, 2020–2026)

| Strategy | Sharpe Ratio | Max Drawdown | Accuracy |
|---|---|---|---|
| Buy & Hold | 0.566 | -33.7% | — |
| SMA(50,200) | 0.362 | -33.7% | — |
| RSI(14,30,50) | -0.148 | -28.3% | — |
| ML (Logistic Regression) | 0.393 | -30.5% | 50.4% |

Parameters for SMA and RSI were selected on the 2010–2019 training period only, then evaluated unchanged on the 2020–2026 test period above.

## Key findings

- **None of the active strategies beat buy-and-hold** on both the raw and risk-adjusted return in the test period. This actually makes sense, since this time period has been a persistent bull market, so strategies cannot outperform a market that is just going up most of the time.
- **SMA(50,200)'s max drawdown exactly matches buy-and-hold's.** This tells us that the 50/200 day crossover method was too slow to react to abrupt crashes, as we had in March 2020. That's why it held the stock throughout that whole drop with the buy-and-hold. This highlights the limitation of these trend-following signals and putting them against fast shocks.
- **RSI spent most of the test period flat.** This strategy relied on the market being "oversold," but in a bullish market, it is really rare for that to happen. Thus, it doesn't invest much in this market, leading to the weakest Sharpe ratio.
- **The ML model's ~50% test accuracy confirms the well-known difficulty of predicting next-day direction** from simple technical features. Even though it wasn't fully accurate, it was able to outperform the other strategies in this test window. This highlights that even though this is like a coin-flip type of classification, it still is able to produce significant results.

## Challenges and debugging notes

A few real bugs came up during this build:

- **Lookahead bias from a missing shift.** My earliest version multiplied a signal by the same day's return instead of the next day's. This hurt RSI worse than SMA in a sneaky way: the day RSI drops below 30 is usually a bad day for the stock, so without the shift I was basically taking credit for the exact drop that caused the buy signal in the first place.
- **A yfinance MultiIndex column structure** Instead of a normal `Close` column, you get something like `('Close', 'SPY')`, which caused a bunch of confusing shape errors down the line (including one where multiplying two things gave me a giant `(n, n)` array instead of a normal list of numbers). I kept patching around it with `.squeeze()` everywhere until I just fixed it once at the very top by flattening the columns right after downloading the data.
- **An indicator warm-up bug produced a fake result.** When I tested the SMA(50,200) strategy on just the test period, the moving averages didn't have enough history yet for the first ~200 days, so the strategy just sat flat for basically all of early-to-mid 2020 — which happened to be exactly when the COVID crash hit. It looked like the strategy was smart enough to dodge the crash (0.58 Sharpe, only -18.8% drawdown!), but it was actually just missing data, not making a real decision. Once I fixed it to calculate the moving averages using the full price history and only cut it down to the test dates afterward, that "advantage" completely disappeared (0.36 Sharpe, -33.7% drawdown — basically no better than buy-and-hold).
- **Early Sharpe ratio calculations were badly miscalibrated** I accidentally used a column of 0s and 1s instead of actual returns at one point, and separately forgot to convert my risk-free rate from annual to daily before using it. Both mistakes gave me Sharpe ratios over 100, which was a pretty obvious sign something was broken.
- **Max drawdown was initially computed incorrectly** I originally just took `(max - min) / min` without caring about which came first, which doesn't actually mean anything. Then even after fixing the formula, I accidentally applied it to daily returns instead of a cumulative return curve, which doesn't really have a "peak" to fall from.
- **The first ML model collapsed to predicting the majority class every time** That's mathematically the same as buy-and-hold, so at first I thought something was broken with my whole pipeline. Turned out the model was just taking the easy way out since slightly more days are "up" days than "down" days in the data. Using `class_weight='balanced'` fixed it and gave me actual mixed predictions instead.

## Limitations

- Doesn't account for transaction costs or slippage.
- Only tested on SPY, no idea if this holds up on other stocks or asset types.
- I only tried a small set of parameter combos on purpose, not a full grid search, so there could be a better combo out there I didn't test.
- The risk-free rate is annualized with simple division instead of geometric compounding; the difference is tiny at current rates, but worth mentioning.

## Running this project

Open the notebook and run all cells top to bottom. Requires `yfinance`, `pandas`, `numpy`, `matplotlib`, and `scikit-learn`.
