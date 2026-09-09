module trading_core(
    input clk,
    input reset,
    input [31:0] price,
    input [31:0] high,
    input [31:0] low,
    input [31:0] close,
    output [1:0] signal
);

wire [31:0] ema_fast;
wire [31:0] ema_slow;
wire [31:0] atr_val;

// EMA FAST
ema ema_fast_inst(
    .clk(clk),
    .reset(reset),
    .price(price),
    .alpha(16'd21845), // ~0.33
    .ema_out(ema_fast)
);

// EMA SLOW
ema ema_slow_inst(
    .clk(clk),
    .reset(reset),
    .price(price),
    .alpha(16'd9362), // smaller alpha
    .ema_out(ema_slow)
);

// ATR
atr atr_inst(
    .clk(clk),
    .reset(reset),
    .high(high),
    .low(low),
    .close(close),
    .atr_out(atr_val)
);

// SIGNAL
crossover cross_inst(
    .clk(clk),
    .ema_fast(ema_fast),
    .ema_slow(ema_slow),
    .signal(signal)
);

endmodule