import os
import pandas as pd
import yfinance as yf

# ===============================
# CONFIG
# ===============================
DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")
FILE_NAME = "nifty_1m.csv"

SYMBOL = "^NSEI"   # NIFTY index
INTERVAL = "1m"
PERIOD = "5d"      # safe limit for 1m data


# ===============================
# LOAD / DOWNLOAD DATA
# ===============================
def get_data():

    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, FILE_NAME)

    # ---------------------------
    # LOAD FROM CACHE
    # ---------------------------
    if os.path.exists(file_path):
        print(f"[DataLoader] Loading cached data from {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    else:
        print("[DataLoader] Downloading data from Yahoo Finance...")

        df = yf.download(
            SYMBOL,
            interval=INTERVAL,
            period=PERIOD,
            progress=False
        )

        df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }, inplace=True)

        df.to_csv(file_path)
        print(f"[DataLoader] Saved data to {file_path}")

    # ---------------------------
    # CLEAN DATA
    # ---------------------------
    df = df.dropna()

    print(f"[DataLoader] Loaded {len(df):,} rows | {df.index.min()} → {df.index.max()}")

    return df