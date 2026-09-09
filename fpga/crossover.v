module crossover(
    input clk,
    input [31:0] ema_fast,
    input [31:0] ema_slow,
    output reg [1:0] signal
);

reg [31:0] prev_fast;
reg [31:0] prev_slow;

always @(posedge clk) begin

    signal <= 2'b00; // HOLD

    if (prev_fast <= prev_slow && ema_fast > ema_slow)
        signal <= 2'b01; // BUY

    else if (prev_fast >= prev_slow && ema_fast < ema_slow)
        signal <= 2'b10; // SELL

    prev_fast <= ema_fast;
    prev_slow <= ema_slow;
end

endmodule