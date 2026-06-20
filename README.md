
**AnalystLab Africa | Week 6 | Batch A**
**Analyst:** Umar Aishat Ajibola

## Business Problem
An investor holding AAPL stock needs to understand historical price behaviour, 
volatility patterns, and trading trends to determine the best times to hold or 
sell — and to manage downside risk effectively.

## Dataset
- **Source:** Yahoo Finance (via yfinance)
- **Ticker:** AAPL (Apple Inc.)
- **Date Range:** June 2021 – June 2026
- **Total Trading Days:** 1,255 rows
- **Features:** Date, Open, High, Low, Close, Volume

## Tools & Libraries
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- yfinance

## Key Findings
- AAPL delivered 131.1% return over 5 years ($128 → $298)
- 2022 was the highest risk period — volatility peaked above 3%
- Best day: +15.33% | Worst day: -9.25% — both in April 2025
- September consistently the weakest month for AAPL returns
- Volume spikes above 65M shares signal major price movements

## Files
| File | Description |
|------|-------------|
| `Aishat_WK6.ipynb` | Main Jupyter Notebook |
| `AAPL_5yr.csv` | Raw dataset |
| `AAPL_5yr_cleaned.csv` | Cleaned dataset |
| `chart1_price_trend.png` | Closing price trend |
| `chart2_volume.png` | Trading volume trend |
| `chart3_volatility.png` | 30-day rolling volatility |
| `chart4_heatmap.png` | Monthly returns heatmap |
| `chart5_distribution.png` | Daily return distribution |
| `Aishat_AAPL_Insight_Report.pdf` | Insight summary report |
