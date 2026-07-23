# Drop the aluminum primary price file here

The model auto-discovers a `date + close` (ideally OHLCV) file in this folder. Name it
`aluminum.csv`. Required format (header row, daily rows, sorted or unsorted both fine):

```
Date,Open,High,Low,Close,Volume
2015-01-02,1796.0,1812.0,1790.0,1805.0,12345
...
```

- **Date** — any parseable date.
- **Close** — REQUIRED. The close-to-close direction of this column is the target.
- **Open/High/Low** — optional but strongly recommended: they enable the candle features
  and, in the morning model, the `next_gap_return` LME-overnight proxy (the US pit opens
  after the LME close, so tomorrow's open already prices in the LME move).
- **Volume** — optional; enables volume features.

See `../../DATA_NEEDED.md` for which series to use and where to get it.
