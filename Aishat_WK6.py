import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Download real AAPL data
aapl = yf.download('AAPL', start='2021-06-20', end='2026-06-20')
aapl.reset_index(inplace=True)
aapl.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
aapl = aapl[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Round to 2 decimal places
aapl['Open'] = aapl['Open'].round(2)
aapl['High'] = aapl['High'].round(2)
aapl['Low'] = aapl['Low'].round(2)
aapl['Close'] = aapl['Close'].round(2)

# Rename to df
df = aapl.copy()

# Save cleaned dataset
df.to_csv('AAPL_5yr_cleaned.csv', index=False)
print("Cleaned dataset saved!")
print(f"Shape: {df.shape}")
print(df.head())

# ============================================================
# STEP 2: DATA EXPLORATION
# ============================================================

print("\n=== STEP 2: DATA EXPLORATION ===")

print("\n--- Column Data Types ---")
print(df.dtypes)

print("\n--- Descriptive Statistics ---")
print(df.describe().round(2))

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Last 5 Rows ---")
print(df.tail())

# ============================================================
# STEP 3: DATA CLEANING & PREPROCESSING
# ============================================================

print("\n=== STEP 3: DATA CLEANING ===")

# Sort by date
df = df.sort_values('Date').reset_index(drop=True)

# Check duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# Check negative prices
print(f"Negative prices: {(df[['Open','High','Low','Close']] < 0).sum().sum()}")

# Validate High >= Low
print(f"Invalid High < Low entries: {(df['High'] < df['Low']).sum()}")

# Ensure Date is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Save cleaned dataset
df.to_csv('AAPL_5yr_cleaned.csv', index=False)

print("Data is clean and ready for analysis!")
print("Cleaned dataset saved as AAPL_5yr_cleaned.csv ✅")

# ============================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================

print("\n=== STEP 4: FEATURE ENGINEERING ===")

# Time features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%b')
df['Quarter'] = df['Date'].dt.quarter
df['Day_of_Week'] = df['Date'].dt.day_name()

# Price features
df['Daily_Price_Change'] = (df['Close'] - df['Open']).round(2)
df['Daily_Price_Change_Pct'] = ((df['Close'] - df['Open']) / df['Open'] * 100).round(4)
df['Price_Range'] = (df['High'] - df['Low']).round(2)
df['Daily_Return'] = (df['Close'].pct_change() * 100).round(4)

# Moving averages
df['MA_7'] = df['Close'].rolling(window=7).mean().round(2)
df['MA_30'] = df['Close'].rolling(window=30).mean().round(2)
df['MA_90'] = df['Close'].rolling(window=90).mean().round(2)

# Volatility
df['Volatility_30'] = df['Daily_Return'].rolling(window=30).std().round(4)

# Monthly returns
df['Monthly_Return'] = df.groupby(
    [df['Date'].dt.year, df['Date'].dt.month])['Close'].transform(
    lambda x: ((x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100).round(2)
)

print("Features created successfully!")
print(f"\nNew columns added: {list(df.columns)}")
print("\n--- Sample with new features ---")
print(df[['Date', 'Close', 'Daily_Return', 'MA_7', 'MA_30', 'Volatility_30']].tail(5))

# ============================================================
# STEP 5: VISUALIZATIONS
# ============================================================

ACCENT = '#00C897'
ACCENT2 = '#FF6B6B'
ACCENT3 = '#FFD93D'
BG = '#0A1F0F'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': BG,
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'axes.titlecolor': 'white',
    'axes.edgecolor': ACCENT,
    'grid.color': '#1a3a2a',
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})

# --- Chart 1: Closing Price Trend with Moving Averages ---
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df['Date'], df['Close'], color='white', linewidth=1, alpha=0.7, label='Close Price')
ax.plot(df['Date'], df['MA_7'], color=ACCENT, linewidth=1.5, label='7-Day MA')
ax.plot(df['Date'], df['MA_30'], color=ACCENT3, linewidth=2, label='30-Day MA')
ax.plot(df['Date'], df['MA_90'], color=ACCENT2, linewidth=2, label='90-Day MA')
ax.fill_between(df['Date'], df['Close'], alpha=0.1, color=ACCENT)
ax.set_title('AAPL Stock Closing Price Trend (2021–2026)', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend(facecolor=BG, edgecolor=ACCENT)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True)
plt.tight_layout()
plt.savefig('chart1_price_trend.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("Chart 1 saved ✅")

# --- Chart 2: Trading Volume Trend ---
fig, ax = plt.subplots(figsize=(15, 5))
colors = [ACCENT if v > df['Volume'].mean() else '#3a7a5a' for v in df['Volume']]
ax.bar(df['Date'], df['Volume'] / 1e6, color=colors, alpha=0.8, width=1.5)
ax.axhline(df['Volume'].mean() / 1e6, color=ACCENT2, linestyle='--',
           linewidth=2, label=f"Avg Volume: {df['Volume'].mean()/1e6:.0f}M")
ax.set_title('AAPL Trading Volume (2021–2026)', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Date')
ax.set_ylabel('Volume (Millions)')
ax.legend(facecolor=BG, edgecolor=ACCENT)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True, axis='y')
plt.tight_layout()
plt.savefig('chart2_volume.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("Chart 2 saved ✅")

# --- Chart 3: 30-Day Rolling Volatility ---
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(df['Date'], df['Volatility_30'], color=ACCENT2, linewidth=1.5)
ax.fill_between(df['Date'], df['Volatility_30'], alpha=0.3, color=ACCENT2)
high_vol = df['Volatility_30'].quantile(0.75)
ax.axhline(high_vol, color=ACCENT3, linestyle='--', linewidth=1.5,
           label=f'High Volatility Threshold: {high_vol:.2f}%')
ax.set_title('AAPL 30-Day Rolling Volatility — Risk Indicator (2021–2026)',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Date')
ax.set_ylabel('Volatility (Std of Daily Returns %)')
ax.legend(facecolor=BG, edgecolor=ACCENT)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.grid(True)
plt.tight_layout()
plt.savefig('chart3_volatility.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("Chart 3 saved ✅")

# --- Chart 4: Monthly Returns Heatmap ---
monthly = df.groupby(['Year', 'Month'])['Daily_Return'].mean().unstack()
month_names = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
monthly.columns = [month_names[m-1] for m in monthly.columns]
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(monthly, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Avg Daily Return (%)'},
            annot_kws={'size': 9})
ax.set_title('AAPL Monthly Average Daily Returns Heatmap (%)',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Month')
ax.set_ylabel('Year')
plt.tight_layout()
plt.savefig('chart4_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("Chart 4 saved ✅")

# --- Chart 5: Daily Return Distribution ---
fig, ax = plt.subplots(figsize=(12, 5))
df['Daily_Return'].dropna().hist(bins=80, ax=ax, color=ACCENT,
                                  edgecolor=BG, alpha=0.9)
ax.axvline(0, color='white', linestyle='--', linewidth=1.5, label='Zero Return')
ax.axvline(df['Daily_Return'].mean(), color=ACCENT3, linestyle='--',
           linewidth=2, label=f"Mean Return: {df['Daily_Return'].mean():.2f}%")
ax.set_title('Distribution of AAPL Daily Returns (2021–2026)',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Daily Return (%)')
ax.set_ylabel('Frequency')
ax.legend(facecolor=BG, edgecolor=ACCENT)
ax.grid(True, axis='y')
plt.tight_layout()
plt.savefig('chart5_distribution.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("Chart 5 saved ✅")