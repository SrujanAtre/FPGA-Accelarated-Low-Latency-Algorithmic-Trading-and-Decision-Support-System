import os
import sys
import struct
import serial
import pandas as pd


# ============================================================
# FPGA DEBUG
#
# FPGA response:
#
# Byte 0-3   : latency cycles
# Byte 4-7   : EMA fast
# Byte 8-11  : EMA slow
# Byte 12-15 : ATR
# Byte 16    : signal
#
# FPGA clock = 125 MHz
# ============================================================


# ============================================================
# CONFIG
# ============================================================

PORT = "COM7"
BAUD = 115200

TARGET_DATE = "2026-03-20"
START_TIME = "09:35"
END_TIME = "09:55"

SCALE = 10000

FPGA_CLOCK_MHZ = 125
CYCLE_NS = 1000 / FPGA_CLOCK_MHZ

SIGNAL_MAP = {
    0x00: "HOLD",
    0x01: "BUY",
    0x02: "SELL"
}


# ============================================================
# LOAD DATA
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from data.data_loader import get_data


print("\n[FPGA DEBUG] Loading market data...")

df = get_data()

required_cols = [
    "open",
    "high",
    "low",
    "close",
    "volume"
]

for col in required_cols:

    if col not in df.columns:
        raise ValueError(
            f"[FPGA ERROR] Missing column: {col}"
        )


# ============================================================
# DATA FREEZE
# ============================================================

df = df.tz_convert("Asia/Kolkata")

df = df[
    df.index.date ==
    pd.to_datetime(TARGET_DATE).date()
]

df = df.between_time(
    START_TIME,
    END_TIME
)

df = df.dropna()


print("\n[DATA FREEZE MATCHED]")
print(f"Date: {TARGET_DATE}")
print(f"Time: {START_TIME} to {END_TIME}")
print(f"Filtered Rows: {len(df)}")


if len(df) < 2:

    raise ValueError(
        "[FPGA ERROR] Not enough rows."
    )


# ============================================================
# PRICE CONVERSION
# ============================================================

def price_to_fpga(value):

    value = float(value)

    scaled = int(
        round(value * SCALE)
    )

    if (
        scaled < -2147483648
        or
        scaled > 2147483647
    ):

        raise ValueError(
            f"[FPGA ERROR] Value {scaled} "
            "does not fit signed 32-bit."
        )

    return scaled


# ============================================================
# BUILD PACKET
# ============================================================

def build_packet(high, low, close):

    high_fpga = price_to_fpga(high)
    low_fpga = price_to_fpga(low)
    close_fpga = price_to_fpga(close)

    packet = bytes([0xAA])

    packet += struct.pack(
        ">i",
        high_fpga
    )

    packet += struct.pack(
        ">i",
        low_fpga
    )

    packet += struct.pack(
        ">i",
        close_fpga
    )

    return packet


# ============================================================
# DECODE DEBUG RESPONSE
# ============================================================

def decode_response(response):

    if len(response) != 17:

        raise ValueError(
            f"[FPGA ERROR] Expected 17 bytes, "
            f"received {len(response)}."
        )

    latency_cycles = struct.unpack(
        ">I",
        response[0:4]
    )[0]

    ema_fast_raw = struct.unpack(
        ">i",
        response[4:8]
    )[0]

    ema_slow_raw = struct.unpack(
        ">i",
        response[8:12]
    )[0]

    atr_raw = struct.unpack(
        ">i",
        response[12:16]
    )[0]

    signal_code = response[16]

    signal = SIGNAL_MAP.get(
        signal_code,
        "UNKNOWN"
    )

    latency_ns = latency_cycles * CYCLE_NS

    ema_fast = ema_fast_raw / SCALE
    ema_slow = ema_slow_raw / SCALE
    atr = atr_raw / SCALE

    return (
        latency_cycles,
        latency_ns,
        ema_fast,
        ema_slow,
        atr,
        signal
    )


# ============================================================
# OPEN SERIAL
# ============================================================

print(
    f"\n[FPGA DEBUG] Opening "
    f"{PORT} at {BAUD} baud..."
)

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=3,
    write_timeout=3
)

ser.reset_input_buffer()
ser.reset_output_buffer()

print(
    "[FPGA DEBUG] Serial connection established."
)


# ============================================================
# INITIALIZATION SAMPLE
# ============================================================

first_row = df.iloc[0]

first_packet = build_packet(
    first_row["high"],
    first_row["low"],
    first_row["close"]
)

print(
    "\n[FPGA DEBUG] Sending "
    "initialization sample..."
)

print(
    "TX:",
    " ".join(
        f"{b:02X}"
        for b in first_packet
    )
)

ser.write(first_packet)
ser.flush()

first_response = ser.read(17)

if len(first_response) != 17:

    ser.close()

    raise RuntimeError(
        "[FPGA ERROR] Did not receive "
        "17-byte initialization response."
    )

print(
    "Initialization response:",
    " ".join(
        f"{b:02X}"
        for b in first_response
    )
)


# ============================================================
# NORMAL SAMPLES
# ============================================================

results = []

print(
    "\n[FPGA DEBUG] Processing "
    "remaining samples..."
)


for i in range(1, len(df)):

    row = df.iloc[i]

    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])

    packet = build_packet(
        high,
        low,
        close
    )

    ser.reset_input_buffer()

    ser.write(packet)
    ser.flush()

    response = ser.read(17)

    if len(response) != 17:

        print(
            f"[FPGA ERROR] Row {i}: "
            f"expected 17 bytes, "
            f"received {len(response)}"
        )

        continue


    (
        latency_cycles,
        latency_ns,
        ema_fast,
        ema_slow,
        atr,
        signal
    ) = decode_response(response)


    results.append({

        "time":
            str(df.index[i]),

        "high":
            high,

        "low":
            low,

        "close":
            close,

        "fpga_ema_fast":
            ema_fast,

        "fpga_ema_slow":
            ema_slow,

        "fpga_atr":
            atr,

        "fpga_latency_cycles":
            latency_cycles,

        "fpga_latency_ns":
            latency_ns,

        "fpga_signal":
            signal

    })


    print(
        f"\n[{i:03d}] "
        f"{df.index[i]}"
    )

    print(
        f"Close       = {close:.2f}"
    )

    print(
        f"EMA Fast    = {ema_fast:.6f}"
    )

    print(
        f"EMA Slow    = {ema_slow:.6f}"
    )

    print(
        f"ATR         = {atr:.6f}"
    )

    print(
        f"Signal      = {signal}"
    )

    print(
        f"Latency     = "
        f"{latency_cycles} cycles "
        f"({latency_ns} ns)"
    )


# ============================================================
# CLOSE
# ============================================================

ser.close()

print(
    "\n[FPGA DEBUG] Serial connection closed."
)


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(
    results
)

output_file = "fpga_debug.csv"

results_df.to_csv(
    output_file,
    index=False
)


print(
    "\n[FPGA DEBUG] Results saved to:"
    f" {output_file}"
)


# ============================================================
# DISPLAY
# ============================================================

print(
    "\n=========================================="
)

print(
    " FPGA DEBUG RESULTS"
)

print(
    "=========================================="
)

print(
    results_df.to_string(
        index=False
    )
)

print(
    "\n[FPGA DEBUG] "
    "Debug measurement completed."
)