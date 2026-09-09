module atr(
    input clk,
    input reset,
    input [31:0] high,
    input [31:0] low,
    input [31:0] close,
    output reg [31:0] atr_out
);

reg [31:0] prev_close;
reg [31:0] tr;

always @(posedge clk or posedge reset) begin
    if (reset) begin
        atr_out <= 0;
        prev_close <= 0;
    end else begin

        // True range
        tr = high - low;

        if ((high > prev_close ? high - prev_close : prev_close - high) > tr)
            tr = (high > prev_close ? high - prev_close : prev_close - high);

        if ((low > prev_close ? low - prev_close : prev_close - low) > tr)
            tr = (low > prev_close ? low - prev_close : prev_close - low);

        // ATR smoothing
        atr_out <= (atr_out * 13 + tr) / 14;

        prev_close <= close;
    end
end

endmodule