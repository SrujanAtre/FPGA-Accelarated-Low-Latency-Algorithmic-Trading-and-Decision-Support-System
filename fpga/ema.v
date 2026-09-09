module ema(
    input clk,
    input reset,
    input [31:0] price,
    input [15:0] alpha,
    output reg [31:0] ema_out
);

reg [31:0] ema_prev;

always @(posedge clk or posedge reset) begin
    if (reset) begin
        ema_prev <= 0;
        ema_out <= 0;
    end else begin
        ema_prev <= ema_prev + ((price - ema_prev) * alpha >> 16);
        ema_out <= ema_prev;
    end
end

endmodule