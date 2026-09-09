# ===============================
# CPU LOW-LATENCY PIPELINE (BENCHMARK)
# ===============================

import sys
import os
import time
import pandas as pd

# -------------------------------
# FIX IMPORT PATH
# -------------------------------
sys.path.append(os.path.abspath(".."))

from data.data_loader import get_data


# ===============================
# LOAD DATA
# ===============================
print("\n[CPU] Loading market data...")
df = get_data()

required_cols = ["open", "high", "low", "close", "volume"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"[CPU ERROR] Missing column: {col}")

print(f"[CPU] Data loaded: {len(df)} rows")


# ===============================
# DATA FREEZE
# ===============================

# Convert UTC → IST
df = df.tz_convert("Asia/Kolkata")

# Set date from Word file
TARGET_DATE = "2026-03-20"

df = df[df.index.date == pd.to_datetime(TARGET_DATE).date()]

# Set time window
START_TIME = "09:35"
END_TIME = "09:55"

df = df.between_time(START_TIME, END_TIME)

print("\n[DATA FREEZE MATCHED]")
print(f"Date: {TARGET_DATE}")
print(f"Time: {START_TIME} to {END_TIME}")
print(f"Filtered Rows: {len(df)}")


# ===============================
# PARAMETERS
# ===============================
FAST_PERIOD = 5
SLOW_PERIOD = 13
ATR_PERIOD = 14

ALPHA_FAST = 2 / (FAST_PERIOD + 1)
ALPHA_SLOW = 2 / (SLOW_PERIOD + 1)


# ===============================
# BENCHMARK SETTINGS
# ===============================
NUM_RUNS = 4

all_latency_values = []
all_run_results = []


# ===============================
# RUN BENCHMARK
# ===============================
print(f"\n[CPU] Running benchmark ({NUM_RUNS} runs)...")

for run_number in range(1, NUM_RUNS + 1):

    # -------------------------------
    # RESET STATE FOR EACH RUN
    # -------------------------------
    ema_fast = df["close"].iloc[0]
    ema_slow = df["close"].iloc[0]
    prev_close = df["close"].iloc[0]
    atr = 0

    prev_ema_fast = ema_fast
    prev_ema_slow = ema_slow

    results = []

    # -------------------------------
    # MAIN LOOP
    # -------------------------------
    for i in range(1, len(df)):

        high = float(df["high"].iloc[i])
        low = float(df["low"].iloc[i])
        close = float(df["close"].iloc[i])

        # Start latency measurement
        start = time.perf_counter_ns()

        # -------------------------------
        # EMA
        # -------------------------------
        ema_fast = ema_fast + ALPHA_FAST * (close - ema_fast)
        ema_slow = ema_slow + ALPHA_SLOW * (close - ema_slow)

        # -------------------------------
        # ATR
        # -------------------------------
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        tr = max(tr1, tr2, tr3)

        atr = (atr * (ATR_PERIOD - 1) + tr) / ATR_PERIOD

        prev_close = close

        # -------------------------------
        # SIGNAL
        # -------------------------------
        signal = "HOLD"

        if prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow:
            signal = "BUY"

        elif prev_ema_fast >= prev_ema_slow and ema_fast < ema_slow:
            signal = "SELL"

        # End latency measurement
        end = time.perf_counter_ns()

        latency = end - start

        # -------------------------------
        # STORE RESULT
        # -------------------------------
        results.append({
            "run": run_number,
            "time": str(df.index[i]),
            "close": close,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "atr": atr,
            "signal": signal,
            "latency_ns": latency
        })

        # Update previous values
        prev_ema_fast = ema_fast
        prev_ema_slow = ema_slow

    # -------------------------------
    # STORE RUN RESULTS
    # -------------------------------
    run_df = pd.DataFrame(results)

    all_run_results.append(run_df)

    # Collect latency values from this run
    all_latency_values.extend(
        run_df["latency_ns"].tolist()
    )

    # -------------------------------
    # DISPLAY THIS RUN
    # -------------------------------
    print(f"\n========== RUN {run_number} ==========")

    print(
        run_df[
            [
                "time",
                "close",
                "ema_fast",
                "ema_slow",
                "atr",
                "signal",
                "latency_ns"
            ]
        ].to_string(index=False)
    )


# ===============================
# COMBINE ALL RUNS
# ===============================
all_results_df = pd.concat(
    all_run_results,
    ignore_index=True
)


# ===============================
# FINAL LATENCY STATISTICS
# ===============================
average_latency = all_results_df["latency_ns"].mean()

minimum_latency = all_results_df["latency_ns"].min()

maximum_latency = all_results_df["latency_ns"].max()


# ===============================
# SAVE OUTPUT
# ===============================
output_file = "latency_log.csv"

all_results_df.to_csv(
    output_file,
    index=False
)


# ===============================
# FINAL OUTPUT
# ===============================
print("\n[CPU] Benchmark completed successfully!")


# ===============================
# FINAL LATENCY STATISTICS
# ===============================
print("\n")
print("==============================================")
print("       FINAL CPU LATENCY STATISTICS")
print("==============================================")

print(f"Number of Runs        : {NUM_RUNS}")
print(f"Samples per Run       : {len(df) - 1}")
print(f"Total Latency Samples : {len(all_latency_values)}")

print("----------------------------------------------")

print(f"Average Latency (ns)  : {average_latency:.2f}")
print(f"Minimum Latency (ns)  : {minimum_latency:.2f}")
print(f"Maximum Latency (ns)  : {maximum_latency:.2f}")

print("==============================================")


# ===============================
# FILE LOCATION
# ===============================
print(f"\n[CPU] Results saved to: {output_file}")