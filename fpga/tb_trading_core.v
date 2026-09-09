module tb_trading_core;

reg clk;
reg reset;

reg [31:0] price;
reg [31:0] high;
reg [31:0] low;
reg [31:0] close;

wire [1:0] signal;

trading_core uut(
    .clk(clk),
    .reset(reset),
    .price(price),
    .high(high),
    .low(low),
    .close(close),
    .signal(signal)
);

// Clock
always #5 clk = ~clk;

initial begin

clk = 0;
reset = 1;

#10 reset = 0;

// Sample data (like CPU pipeline)
price = 26000; high = 26100; low = 25900; close = 26000;
#10;

price = 26200; high = 26300; low = 26100; close = 26200;
#10;

price = 25800; high = 25900; low = 25700; close = 25800;
#10;

price = 26500; high = 26600; low = 26400; close = 26500;
#50;

$finish;

end

endmodule