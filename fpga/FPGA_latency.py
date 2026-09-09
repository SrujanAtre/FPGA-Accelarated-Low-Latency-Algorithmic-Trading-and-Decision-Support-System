import os
import sys
import time
import struct
import serial
import pandas as pd

# ============================================================
# FPGA LATENCY MEASUREMENT
#
# Uses the same frozen Yahoo Finance OHLC data as the CPU
# pipeline and sends the samples sequentially to the ZCU104.
#
# FPGA response:
#   Byte 0-3 : latency in FPGA clock cycles
#   Byte 4   : signal
#
# FPGA clock = 125 MHz
# 1 clock cycle = 8 ns
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

print("\n[FPGA] Loading market data...")
df = get_data()

required_cols = ["open", "high", "low", "close", "volume"]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"[FPGA ERROR] Missing column: {col}")


# ============================================================
# DATA FREEZE
# ============================================================

df = df.tz_convert("Asia/Kolkata")

df = df[df.index.date == pd.to_datetime(TARGET_DATE).date()]

df = df.between_time(START_TIME, END_TIME)

df = df.dropna()

print("\n[DATA FREEZE MATCHED]")
print(f"Date: {TARGET_DATE}")
print(f"Time: {START_TIME} to {END_TIME}")
print(f"Filtered Rows: {len(df)}")


if len(df) < 2:
    raise ValueError("[FPGA ERROR] Not enough rows in frozen dataset.")


# ============================================================
# CONVERT PRICE TO FPGA FIXED-POINT FORMAT
# ============================================================

def price_to_fpga(value):
    """
    FPGA uses signed 32-bit integers with SCALE = 10000.
    Example:
        26003.90 -> 260039000
    """
    value = float(value)
    scaled = int(round(value * SCALE))

    if scaled < -2147483648 or scaled > 2147483647:
        raise ValueError(
            f"[FPGA ERROR] Scaled value {scaled} does not fit "
            "in signed 32-bit integer."
        )

    return scaled


# ============================================================
# BUILD FPGA PACKET
# ============================================================

def build_packet(high, low, close):
    """
    Packet format:

    AA
    HIGH  : 4 bytes, MSB first
    LOW   : 4 bytes, MSB first
    CLOSE : 4 bytes, MSB first
    """

    high_fpga = price_to_fpga(high)
    low_fpga = price_to_fpga(low)
    close_fpga = price_to_fpga(close)

    packet = bytes([0xAA])

    packet += struct.pack(">i", high_fpga)
    packet += struct.pack(">i", low_fpga)
    packet += struct.pack(">i", close_fpga)

    return packet


# ============================================================
# DECODE FPGA RESPONSE
# ============================================================

def decode_response(response):
    """
    FPGA response:

    Byte 0-3 : latency cycles, unsigned 32-bit MSB first
    Byte 4   : signal
    """

    if len(response) != 5:
        raise ValueError(
            f"[FPGA ERROR] Expected 5 response bytes, "
            f"received {len(response)}."
        )

    latency_cycles = struct.unpack(">I", response[0:4])[0]

    signal_code = response[4]

    signal = SIGNAL_MAP.get(signal_code, "UNKNOWN")

    latency_ns = latency_cycles * CYCLE_NS

    return latency_cycles, latency_ns, signal


# ============================================================
# OPEN SERIAL PORT
# ============================================================

print(f"\n[FPGA] Opening {PORT} at {BAUD} baud...")

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=2,
    write_timeout=2
)

# Remove old data from previous tests.
ser.reset_input_buffer()
ser.reset_output_buffer()

print("[FPGA] Serial connection established.")


# ============================================================
# SEND FIRST SAMPLE
#
# The FPGA needs the first sample to initialize:
#   EMA fast
#   EMA slow
#   ATR previous close
#   crossover state
#
# The CPU pipeline also uses the first row only for
# initialization.
# ============================================================

first_row = df.iloc[0]

first_packet = build_packet(
    first_row["high"],
    first_row["low"],
    first_row["close"]
)

print("\n[FPGA] Sending initialization sample...")
print("TX:", " ".join(f"{b:02X}" for b in first_packet))

ser.write(first_packet)
ser.flush()

# The FPGA will also produce a response for this sample.
# Read and discard it because this is the initialization row.
first_response = ser.read(5)

if len(first_response) != 5:
    ser.close()
    raise RuntimeError(
        "[FPGA ERROR] Did not receive the 5-byte initialization response."
    )

print(
    "Initialization response:",
    " ".join(f"{b:02X}" for b in first_response)
)


# ============================================================
# PROCESS ALL NORMAL SAMPLES
# ============================================================

results = []

print("\n[FPGA] Processing remaining OHLC samples...")


for i in range(1, len(df)):

    row = df.iloc[i]

    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])

    packet = build_packet(high, low, close)

    # --------------------------------------------------------
    # Send the exact OHLC row to the FPGA.
    # --------------------------------------------------------

    ser.reset_input_buffer()

    ser.write(packet)
    ser.flush()

    # --------------------------------------------------------
    # Receive exactly 5 bytes:
    #
    # latency[31:24]
    # latency[23:16]
    # latency[15:8]
    # latency[7:0]
    # signal
    # --------------------------------------------------------

    response = ser.read(5)

    if len(response) != 5:
        print(
            f"[FPGA ERROR] Row {i}: expected 5 bytes, "
            f"received {len(response)}"
        )
        continue

    latency_cycles, latency_ns, signal = decode_response(response)

    results.append({
        "time": str(df.index[i]),
        "high": high,
        "low": low,
        "close": close,
        "fpga_latency_cycles": latency_cycles,
        "fpga_latency_ns": latency_ns,
        "fpga_signal": signal
    })

    print(
        f"[{i:03d}] "
        f"{df.index[i]} | "
        f"Close={close:.2f} | "
        f"Latency={latency_cycles} cycles "
        f"({latency_ns} ns) | "
        f"Signal={signal}"
    )


# ============================================================
# CLOSE SERIAL PORT
# ============================================================

ser.close()

print("\n[FPGA] Serial connection closed.")


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_file = "fpga_latency.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("\n[FPGA] Latency measurement completed.")


# ============================================================
# DISPLAY RESULTS
# ============================================================

if len(results_df) > 0:

    print("\n======================================")
    print(" FPGA LATENCY RESULTS")
    print("======================================")

    print(
        f"Samples processed       : "
        f"{len(results_df)}"
    )

    print(
        f"Average Latency (ns)   : "
        f"{results_df['fpga_latency_ns'].mean():.2f}"
    )

    print(
        f"Minimum Latency (ns)   : "
        f"{results_df['fpga_latency_ns'].min():.2f}"
    )

    print(
        f"Maximum Latency (ns)   : "
        f"{results_df['fpga_latency_ns'].max():.2f}"
    )

    print(
        f"Average Latency (cycles): "
        f"{results_df['fpga_latency_cycles'].mean():.2f}"
    )

    print("\nSignal Counts:")
    print(results_df["fpga_signal"].value_counts())

    print("\nSample FPGA Results:")
    print(results_df.head())

    print(f"\n[FPGA] Results saved to: {output_file}")

else:

    print("\n[FPGA ERROR] No valid FPGA results were collected.")